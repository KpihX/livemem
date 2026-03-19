"""
memory.py — Main LiveMem orchestrator.

WHY a single orchestrator class:
    LiveMem owns the graph, the tiered index, the embedder, and all
    algorithmic logic (awake ingest, sleep consolidation, retrieval).
    A single class keeps these tightly coupled components coherent and
    avoids the "distributed state" problem where multiple objects each
    own a partial view of the memory.

    The AWAKE / SLEEP split follows neuroscience:
    - AWAKE: fast local operations only (SHORT tier ANN, DIRECT edges).
    - SLEEP: background diffusion across tiers, promotion, compression.
      This is intentionally slower and more expensive — it runs during
      idle periods via SleepDaemon.
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np

from livemem.config import DEFAULT_CONFIG, LiveConfig
from livemem.embedder import BaseEmbedder, CrossEncoderReranker, make_embedder
from livemem.graph import Graph
from livemem.index import TieredIndex
from livemem.types import (
    Edge,
    EdgeType,
    Node,
    RetrievalResult,
    Tier,
    strength_effective,
    urgency_effective,
    tier_fn,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class LiveMem:
    """Brain-inspired tiered graph memory.

    Architecture
    ------------
    Three memory tiers (SHORT / MEDIUM / LONG) backed by:
      - A directed Graph of Node objects (V) and Edge objects (E/E_r).
      - A TieredIndex (3× hnswlib HNSW) for approximate ANN search.
      - An Embedder (mock or real fastembed) for text → vector.

    AWAKE mode  (ingest_awake):
        Embeds the incoming summary, queries the SHORT index for
        semantically similar existing nodes, creates DIRECT edges (newer →
        older), and adds the node to the SHORT tier.

    SLEEP mode  (sleep_phase):
        Called by SleepDaemon after idle_ttl seconds of no ingestion.
        Runs four passes in order:
          1. _decay_pass      — materialise exponential decay into s_base.
          2. sleep_diffuse    — wire SHORT nodes to MEDIUM/LONG via SLEEP edges.
          3. sleep_promote    — reinforce LONG/MEDIUM nodes evoked by
                               recent activity.
          4. sleep_compress   — fuse very similar LONG nodes into one.

    Retrieval (retrieve):
        Multi-signal scoring: direct cosine + graph traversal + strength +
        importance. Reinforces returned nodes (spacing effect).
    """

    def __init__(
        self,
        cfg: LiveConfig = DEFAULT_CONFIG,
        embedder: BaseEmbedder | None = None,
        mock: bool = False,
    ) -> None:
        """Initialise LiveMem.

        Parameters
        ----------
        cfg      : LiveConfig — all tunable parameters.
        embedder : optional pre-built BaseEmbedder instance.
        mock     : if True and embedder is None, use MockEmbedder.
        """
        self._cfg = cfg
        self._embedder: BaseEmbedder = (
            embedder if embedder is not None else make_embedder(cfg, mock=mock)
        )
        self._graph = Graph()
        self._index = TieredIndex(cfg)
        self._last_sleep_end: float = 0.0
        # Cross-encoder re-ranker — lazy-loaded on first retrieve() call
        # when cfg.reranker_enabled is True. Always instantiated so the
        # object is ready; model download is deferred until rerank().
        self._reranker = CrossEncoderReranker(cfg)

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def graph(self) -> Graph:
        return self._graph

    @property
    def index(self) -> TieredIndex:
        return self._index

    @property
    def cfg(self) -> LiveConfig:
        return self._cfg

    @property
    def last_sleep_end(self) -> float:
        return self._last_sleep_end

    @last_sleep_end.setter
    def last_sleep_end(self, value: float) -> None:
        self._last_sleep_end = value

    # ── Awake ingestion ────────────────────────────────────────────────────────

    def ingest_awake(
        self,
        summary: str,
        ref_uri: str | None = None,
        ref_type: str = "text",
        importance: float = 0.5,
        urgency: float = 0.0,
    ) -> str:
        """Ingest a new memory unit during an active (awake) session.

        Algorithm
        ---------
        1. Embed summary → unit-norm vector v.
        2. Query SHORT ANN index for k_awake neighbours (BEFORE adding
           the new node so we never self-link).
        3. For each neighbour with cos ≥ theta_min, create a DIRECT edge
           from new_node → neighbour (newer → older, enforced by t order).
        4. Add node to graph + SHORT index.
        5. Return the new node's UUID.

        WHY only query SHORT during awake:
            Querying all three tiers at ingest time would be expensive and
            semantically wrong — we only want to link to recent context.
            Cross-tier links are established during the sleep phase.

        Parameters
        ----------
        summary    : str   — short textual summary (≤ 200 chars recommended).
        ref_uri    : str   — path or URL to actual content (None = inline).
        ref_type   : str   — RefType constant.
        importance : float — semantic weight ∈ [0, 1].
        urgency    : float — time-pressure ∈ [0, 1]. Decays from creation time.

        Returns
        -------
        str — UUID of the newly created node.
        """
        now = time.time()
        v = self._embedder.embed(summary)

        # Query SHORT index BEFORE adding the new node.
        candidates = self._index.query(Tier.SHORT, v, self._cfg.k_awake)

        # s_base scales with importance: lerp(s_base_init, 1.0, importance).
        # WHY: a critical fact (imp=1.0) should start at full strength so it
        # stays in SHORT longer and outranks trivial nodes in retrieval.
        # A background fact (imp=0.0) still gets the floor so it can form
        # edges before drifting to LONG over the next sleep cycle.
        s_base = self._cfg.s_base_init + importance * (1.0 - self._cfg.s_base_init)
        node = Node(
            v=v,
            summary=summary,
            ref_uri=ref_uri,
            ref_type=ref_type,
            importance=importance,
            urgency=urgency,
            s_base=s_base,
            t=now,
            t_accessed=now,
            tier=Tier.SHORT,
        )

        # Create DIRECT edges to similar existing SHORT nodes.
        for nb_id, cos in candidates:
            if cos < self._cfg.theta_min:
                continue
            if nb_id not in self._graph:
                continue
            nb_node = self._graph.V[nb_id]
            # Enforce direction: newer (higher t) → older (lower t).
            if node.t >= nb_node.t:
                from_id, to_id = node.id, nb_id
                delta_t = node.t - nb_node.t
            else:
                from_id, to_id = nb_id, node.id
                delta_t = nb_node.t - node.t
            edge = Edge(
                from_id=from_id,
                to_id=to_id,
                cos_sim=cos,
                delta_t=delta_t,
                edge_type=EdgeType.DIRECT,
            )
            self._graph.add_edge_if_new(edge)

        self._graph.add_node(node)
        self._index.add(node.id, v, Tier.SHORT)

        logger.debug("ingest_awake: %s → %s", node.id[:8], summary[:50])
        return node.id

    # ── Sleep phase ────────────────────────────────────────────────────────────

    def sleep_phase(self, idle_duration: float = 0.0) -> None:
        """Run a full sleep cycle: decay → diffuse → promote → compress.

        WHY this order:
            1. _decay_pass first: materialises current s_base values so
               subsequent tier moves use up-to-date strength.
            2. sleep_diffuse: builds cross-tier SLEEP edges from SHORT
               nodes that haven't been diffused yet.
            3. sleep_promote: reinforces LONG/MEDIUM nodes evoked by
               recent SHORT activity, pulling relevant long-term memories
               back toward SHORT.
            4. sleep_compress: fuses similar LONG nodes only after all
               edges have been updated — avoids compressing nodes that
               were just promoted.

        Parameters
        ----------
        idle_duration : float — seconds the system has been idle. Used by
                        sleep_diffuse to decide whether to bridge SHORT→LONG.
        """
        logger.info("sleep_phase start (idle=%.0fs)", idle_duration)
        self._decay_pass()
        self.sleep_diffuse(idle_duration)
        self.sleep_promote()
        self.sleep_compress()
        self._last_sleep_end = time.time()
        logger.info("sleep_phase complete")

    def sleep_diffuse(self, idle_duration: float) -> None:
        """Wire undiffused SHORT nodes to MEDIUM (and optionally LONG) nodes.

        Algorithm
        ---------
        For each SHORT node n that has not yet been diffused:
          - Query MEDIUM index for k_sleep neighbours with cos ≥ theta_sleep.
            → Add SLEEP edge (n → m) if new; reinforce m.
          - If idle_duration ≥ tau_long:
            → Also query LONG index and apply the same pattern.
          - Mark n.diffused = True.

        WHY tau_long guard:
            Bridging SHORT→LONG during an active session would prematurely
            consolidate working memory into semantic storage. Only after
            tau_long seconds of inactivity do we consider the session
            "at rest" and allow full diffusion.
        """
        pending = [
            n for n in self._graph.nodes_in_tier(Tier.SHORT)
            if not n.diffused
        ]
        for n in pending:
            # Diffuse to MEDIUM.
            med_candidates = self._index.query(
                Tier.MEDIUM, n.v, self._cfg.k_sleep
            )
            for m_id, cos in med_candidates:
                if cos < self._cfg.theta_sleep:
                    continue
                if m_id not in self._graph:
                    continue
                m_node = self._graph.V[m_id]
                edge = Edge(
                    from_id=n.id,
                    to_id=m_id,
                    cos_sim=cos,
                    delta_t=abs(n.t - m_node.t),
                    edge_type=EdgeType.SLEEP,
                )
                self._graph.add_edge_if_new(edge)
                self._reinforce(m_node)

            # Diffuse to LONG only after extended idle.
            if idle_duration >= self._cfg.tau_long:
                long_candidates = self._index.query(
                    Tier.LONG, n.v, self._cfg.k_sleep
                )
                for l_id, cos in long_candidates:
                    if cos < self._cfg.theta_sleep:
                        continue
                    if l_id not in self._graph:
                        continue
                    l_node = self._graph.V[l_id]
                    edge = Edge(
                        from_id=n.id,
                        to_id=l_id,
                        cos_sim=cos,
                        delta_t=abs(n.t - l_node.t),
                        edge_type=EdgeType.SLEEP,
                    )
                    self._graph.add_edge_if_new(edge)
                    self._reinforce(l_node)

            n.diffused = True

    def sleep_promote(self) -> None:
        """Reinforce LONG/MEDIUM nodes that are thematically linked to recent activity.

        Algorithm
        ---------
        1. evoked_set = nodes accessed after last_sleep_end OR created
           after last_sleep_end (i.e., active since last sleep).
        2. If empty: return early (nothing to promote from).
        3. Compute evoked_centroid: strength-weighted mean of evoked
           node vectors, then L2-normalise.
        4. Query LONG index with centroid for k_promote neighbours.
           For each with cos ≥ theta_promote:
             → reinforce the LONG node.
             → create SLEEP edge from the most similar evoked node → long node.
        5. Repeat for MEDIUM index.

        WHY a centroid:
            Instead of k_promote queries per evoked node (expensive), one
            centroid query captures the "theme" of recent activity with a
            single ANN lookup.
        """
        now = time.time()

        # Collect nodes active since last sleep.
        evoked = [
            n for n in self._graph.V.values()
            if n.t_accessed >= self._last_sleep_end
            or n.t >= self._last_sleep_end
        ]
        if not evoked:
            logger.debug("sleep_promote: empty evoked set, skipping")
            return

        # Compute strength+urgency-weighted centroid.
        # WHY include_urgency=True: the promote centroid should lean toward
        # topics the user was urgently focused on — pulling long-term memories
        # related to pressing concerns back into SHORT/MEDIUM.
        centroid = self._compute_centroid(evoked, now, include_urgency=True)

        # Helper to find the most similar evoked node to a candidate.
        def most_similar_evoked(candidate_v: np.ndarray) -> Node:
            best_n, best_cos = evoked[0], -1.0
            for en in evoked:
                c = float(np.dot(en.v, candidate_v))
                if c > best_cos:
                    best_cos, best_n = c, en
            return best_n

        # Promote from LONG.
        long_candidates = self._index.query(
            Tier.LONG, centroid, self._cfg.k_promote
        )
        for l_id, cos in long_candidates:
            if cos < self._cfg.theta_promote:
                continue
            if l_id not in self._graph:
                continue
            l_node = self._graph.V[l_id]
            self._reinforce(l_node)
            src = most_similar_evoked(l_node.v)
            edge = Edge(
                from_id=src.id,
                to_id=l_id,
                cos_sim=float(np.dot(src.v, l_node.v)),
                delta_t=abs(src.t - l_node.t),
                edge_type=EdgeType.SLEEP,
            )
            self._graph.add_edge_if_new(edge)

        # Promote from MEDIUM.
        med_candidates = self._index.query(
            Tier.MEDIUM, centroid, self._cfg.k_promote
        )
        for m_id, cos in med_candidates:
            if cos < self._cfg.theta_promote:
                continue
            if m_id not in self._graph:
                continue
            m_node = self._graph.V[m_id]
            self._reinforce(m_node)
            src = most_similar_evoked(m_node.v)
            edge = Edge(
                from_id=src.id,
                to_id=m_id,
                cos_sim=float(np.dot(src.v, m_node.v)),
                delta_t=abs(src.t - m_node.t),
                edge_type=EdgeType.SLEEP,
            )
            self._graph.add_edge_if_new(edge)

    def greedy_cluster(self, tier: Tier, theta: float) -> list[set[str]]:
        """Greedily partition nodes in a tier into similarity clusters.

        Algorithm (single-pass greedy, oldest-first)
        -------------------------------------------
        1. nodes = nodes_in_tier(tier) [oldest first].
        2. unassigned = all node ids in tier.
        3. While unassigned is not empty:
           a. Pick the oldest unassigned node as seed.
           b. Query the tier ANN index for top-50 neighbours of seed.v.
           c. Initialise cluster = {seed}, running centroid = seed.v.
           d. For each ANN candidate (by descending cos):
              - Compute cos(candidate.v, running_centroid).
              - If ≥ theta AND candidate is unassigned: add to cluster,
                update running centroid (mean of member vectors, normalised).
           e. Append cluster to result.
        4. Return list of clusters.

        WHY greedy (not k-means or DBSCAN):
            We run compression inside a sleep phase with limited budget.
            Greedy is O(n · k_query) — deterministic, fast, and produces
            semantically coherent clusters as long as theta is high (≥0.90).
        """
        nodes = self._graph.nodes_in_tier(tier)
        if not nodes:
            return []

        unassigned: set[str] = {n.id for n in nodes}
        clusters: list[set[str]] = []

        while unassigned:
            # Pick oldest unassigned node as seed.
            seed = next(
                n for n in self._graph.nodes_in_tier(tier)
                if n.id in unassigned
            )
            cluster: set[str] = {seed.id}
            unassigned.discard(seed.id)
            centroid = seed.v.copy()

            # Query ANN for neighbours (top 50 or fewer live items).
            k_query = min(50, self._index.size(tier))
            if k_query > 1:
                candidates = self._index.query(tier, centroid, k_query)
                for cand_id, _cos in candidates:
                    if cand_id not in unassigned:
                        continue
                    if cand_id not in self._graph:
                        continue
                    cand_node = self._graph.V[cand_id]
                    cos_with_centroid = float(np.dot(cand_node.v, centroid))
                    if cos_with_centroid >= theta:
                        cluster.add(cand_id)
                        unassigned.discard(cand_id)
                        # Update running centroid: mean of member vectors, normalised.
                        members_v = np.stack(
                            [self._graph.V[mid].v for mid in cluster
                             if mid in self._graph.V]
                        )
                        centroid = members_v.mean(axis=0)
                        norm = np.linalg.norm(centroid)
                        if norm > 1e-12:
                            centroid /= norm

            clusters.append(cluster)

        return clusters

    def sleep_compress(self) -> None:
        """Fuse very similar LONG nodes into consolidated nodes.

        Trigger condition:
            Only runs when LONG tier size ≥ long_compress_fraction * max_nodes.
            This prevents compression from running unnecessarily on a nearly
            empty graph.

        Algorithm (per cluster of size ≥ 2):
        1. Compute strength-weighted centroid vector v_f (normalised).
        2. t_oldest = min t across members (fused node inherits oldest birth).
        3. max_importance, max_s_base from members.
        4. fused_summary = truncated join of member summaries.
        5. Create consolidated Node in LONG.
        6. Collect external neighbours (nodes connected to any cluster member
           but NOT in the cluster themselves).
        7. Remove all cluster nodes from graph + LONG index.
        8. Add fused node.
        9. Reconnect external neighbours: create CONSOLIDATED edge if
           cos(ext.v, v_f) ≥ theta_min (direction by timestamp).
        """
        long_size = self._graph.tier_size(Tier.LONG)
        threshold = int(self._cfg.max_nodes * self._cfg.long_compress_fraction)
        if long_size < threshold:
            logger.debug(
                "sleep_compress: LONG=%d < threshold=%d, skipping",
                long_size, threshold,
            )
            return

        logger.info(
            "sleep_compress: LONG=%d ≥ threshold=%d, clustering...",
            long_size, threshold,
        )
        clusters = self.greedy_cluster(Tier.LONG, self._cfg.theta_compress)
        now = time.time()

        for cluster_ids in clusters:
            if len(cluster_ids) < 2:
                continue  # Singleton — nothing to fuse.

            members = [
                self._graph.V[mid] for mid in cluster_ids
                if mid in self._graph.V
            ]
            if len(members) < 2:
                continue

            # Compute strength-weighted centroid.
            v_f = self._compute_centroid(members, now)

            t_oldest = min(m.t for m in members)
            max_importance = max(m.importance for m in members)
            max_s_base = max(m.s_base for m in members)

            # Build fused summary (truncate each member to 60 chars).
            parts = [m.summary[:60] for m in members]
            fused_summary = "; ".join(parts)
            if len(fused_summary) > 200:
                fused_summary = fused_summary[:197] + "..."

            # Collect external neighbours BEFORE removing cluster nodes.
            external: set[str] = set()
            for mid in cluster_ids:
                for edge in self._graph.E.get(mid, []):
                    if edge.to_id not in cluster_ids:
                        external.add(edge.to_id)
                for edge in self._graph.E_r.get(mid, []):
                    if edge.from_id not in cluster_ids:
                        external.add(edge.from_id)

            # Remove cluster nodes.
            for mid in list(cluster_ids):
                self._graph.remove_node(mid)
                # remove from LONG index (it was already removed by remove_node cascade
                # but we also need to remove from the tier index explicitly).
                # Note: graph.remove_node does NOT touch the index — we do it here.
                self._index.remove(mid, Tier.LONG)

            # Create consolidated node.
            # WHY urgency=0.0: a fused/archived node is by definition not
            # urgent — it represents a stable semantic memory, not a
            # pending task. Urgency would have decayed to near-zero before
            # compression runs (tau_long idle + sleep phase elapsed).
            fused = Node(
                v=v_f,
                summary=fused_summary,
                ref_uri=None,
                ref_type="text",
                importance=max_importance,
                urgency=0.0,
                s_base=max_s_base,
                t=t_oldest,
                t_accessed=now,
                tier=Tier.LONG,
                consolidated=True,
                sources=list(cluster_ids),
            )
            self._graph.add_node(fused)
            self._index.add(fused.id, v_f, Tier.LONG)

            # Reconnect external neighbours.
            for ext_id in external:
                if ext_id not in self._graph:
                    continue
                ext_node = self._graph.V[ext_id]
                cos = float(np.dot(ext_node.v, v_f))
                if cos < self._cfg.theta_min:
                    continue
                # Direction: newer → older.
                if ext_node.t >= fused.t:
                    from_id, to_id = ext_id, fused.id
                    delta_t = ext_node.t - fused.t
                else:
                    from_id, to_id = fused.id, ext_id
                    delta_t = fused.t - ext_node.t
                edge = Edge(
                    from_id=from_id,
                    to_id=to_id,
                    cos_sim=cos,
                    delta_t=delta_t,
                    edge_type=EdgeType.CONSOLIDATED,
                )
                self._graph.add_edge_if_new(edge)

        logger.info("sleep_compress: done")

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def retrieve(
        self, query_text: str, k: int = 10
    ) -> list[RetrievalResult]:
        """Retrieve the top-k most relevant memory nodes for a query.

        Scoring formula (per node n, weights sum to 1.0):
            score(n) = alpha   * cos(q, n.v)
                     + beta    * traversal_score.get(n.id, 0)
                     + gamma   * strength_effective(n, now)
                     + delta   * n.importance          # already ∈ [0,1]
                     + epsilon * urgency_effective(n, now)

        where traversal_score is accumulated by 1-hop graph expansion
        from the top-5 seed nodes (ANN hits + high-importance nodes).

        When cfg.reranker_enabled is True, steps 6-8 change:
            6b. Collect top-reranker_k candidates (instead of top-k).
            7b. Cross-encoder re-ranks (query, summary) pairs → new scores.
            8b. Sort by cross-encoder score, take top-k.

        Algorithm
        ---------
        1. q = embed(query_text).
        2. direct = query SHORT index for k neighbours.
        3. High-importance sweep: query MEDIUM and LONG for k//2 each,
           filtered to importance ≥ importance_medium_floor. This ensures
           semantically important memories are surfaced regardless of tier.
        3b.Urgency sweep: query SHORT and MEDIUM for k//2 each, filtered to
           u_eff ≥ theta_urgent. Guarantees that urgency-pinned nodes always
           enter the candidate pool regardless of cosine similarity to q.
        4. Limit seeds to k (cosine-ranked).
        5. Urgency sweep: scan ALL nodes, collect ids with u_eff ≥ theta_urgent
           into urgent_forced. These bypass the k-limit and are merged into
           all_candidates unconditionally.
        6. 1-hop graph expansion from top-5 seeds.
        7. Build all_candidates = seeds ∪ traversal ∪ urgent_forced.
        8. Compute final_score for all candidates (5 components).
        9. Sort descending, take top-k (or top-reranker_k if reranker on).
        9b.[reranker] Cross-encoder rerank → re-sort → take top-k.
        10. Reinforce all top-k nodes.
        11. Return list[RetrievalResult].

        WHY include high-importance nodes from all tiers:
            A highly important node may have aged into LONG but still
            represents a critical memory. We always surface it regardless
            of tier, ensuring the Eisenhower "important" quadrant is
            never permanently buried.
        """
        now = time.time()
        if self._graph.total_nodes() == 0:
            return []

        q = self._embedder.embed(query_text)

        # ── 1. Direct ANN in SHORT ────────────────────────────────────────────
        direct = self._index.query(Tier.SHORT, q, k)
        seeds: dict[str, float] = {}  # node_id → cos_direct
        for nid, cos in direct:
            if nid in self._graph:
                seeds[nid] = max(seeds.get(nid, 0.0), cos)

        # ── 2. High-importance sweep across MEDIUM and LONG ──────────────────
        k_imp = max(1, k // 2)
        for tier in (Tier.MEDIUM, Tier.LONG):
            for nid, cos in self._index.query(tier, q, k_imp):
                if nid not in self._graph:
                    continue
                node = self._graph.V[nid]
                if node.importance >= self._cfg.importance_medium_floor:
                    seeds[nid] = max(seeds.get(nid, 0.0), cos)

        # Limit seeds to k (cosine-ranked). Urgency-pinned nodes are exempt —
        # they are collected separately below so they are never trimmed out.
        if len(seeds) > k:
            top_seeds = sorted(seeds.items(), key=lambda x: x[1], reverse=True)[:k]
            seeds = dict(top_seeds)

        # ── 3. Urgency sweep — direct graph scan, bypasses seed limit ─────────
        # WHY direct graph scan (not ANN):
        #     ANN query ranks by cosine similarity. An urgent node may be
        #     semantically unrelated to q — it would never appear in the top-k
        #     cosine results. Routing urgency through ANN would silently miss it.
        #     Direct scan is O(N_urgent) not O(N): urgency decays fast
        #     (urgency_lambda=5e-5 → half-life ~4h), so very few nodes satisfy
        #     u_eff ≥ theta_urgent at any time. The scan cost is negligible.
        #     WHY bypass the seed limit:
        #     The seed pool is capped at k (cosine-ranked) above. If we added
        #     urgent nodes to seeds before trimming, a low-cosine urgent node
        #     would be dropped. Instead we collect them into `urgent_forced`
        #     which is merged into all_candidates unconditionally — guaranteeing
        #     every urgency-pinned node always enters the scoring stage.
        urgent_forced: set[str] = set()
        for node in self._graph.V.values():
            u_eff = urgency_effective(node, now, self._cfg)
            if u_eff >= self._cfg.theta_urgent:
                urgent_forced.add(node.id)

        # ── 4. 1-hop graph traversal from top-5 seeds ────────────────────────
        traversal: dict[str, float] = {}
        top5 = sorted(seeds.items(), key=lambda x: x[1], reverse=True)[:5]
        for seed_id, seed_cos in top5:
            for edge in self._graph.E.get(seed_id, []):
                nb_id = edge.to_id
                if nb_id not in self._graph:
                    continue
                nb = self._graph.V[nb_id]
                score_contrib = (
                    seed_cos
                    * edge.cos_sim
                    * strength_effective(nb, now, self._cfg)
                )
                traversal[nb_id] = traversal.get(nb_id, 0.0) + score_contrib

        # ── 5. Build candidate pool (seeds + traversal + urgent_forced) ─────────
        # urgent_forced nodes bypass the seed cosine-rank filter (step 3 above).
        all_candidates: set[str] = set(seeds.keys()) | set(traversal.keys()) | urgent_forced

        # ── 5. Score all candidates (5-component formula) ─────────────────────
        scored: list[tuple[float, str]] = []
        for nid in all_candidates:
            if nid not in self._graph:
                continue
            n = self._graph.V[nid]
            cos_direct = float(np.dot(q, n.v))
            s_eff = strength_effective(n, now, self._cfg)
            u_eff = urgency_effective(n, now, self._cfg)
            trav = traversal.get(nid, 0.0)
            final = (
                self._cfg.alpha_score * cos_direct
                + self._cfg.beta_score * trav
                + self._cfg.gamma_score * s_eff
                + self._cfg.delta_score * n.importance
                + self._cfg.epsilon_score * u_eff
            )
            scored.append((final, nid))

        # Sort descending by 5-component score.
        scored.sort(reverse=True)

        # ── 6. Optional cross-encoder re-ranking ──────────────────────────────
        if self._cfg.reranker_enabled:
            # Expand the pool to reranker_k candidates, then re-rank them.
            # WHY expand: the bi-encoder may rank the true best result at
            # position 8 (just outside top-k). Feeding reranker_k > k into
            # the cross-encoder lets it promote it to the top.
            pool_size = max(k, self._cfg.reranker_k)
            pool = scored[:pool_size]
            # Build (node_id, summary) pairs for the cross-encoder.
            ce_candidates: list[tuple[str, str]] = []
            for _score, nid in pool:
                if nid in self._graph:
                    ce_candidates.append((nid, self._graph.V[nid].summary))
            # Cross-encoder returns (ce_score, node_id) sorted descending.
            reranked = self._reranker.rerank(query_text, ce_candidates)
            # Rebuild top_k as (ce_score, node_id) for the final slice.
            top_k: list[tuple[float, str]] = reranked[:k]
            logger.debug(
                "Cross-encoder reranked %d candidates → top-%d.",
                len(ce_candidates),
                k,
            )
        else:
            top_k = scored[:k]

        # ── 7. Reinforce returned nodes ───────────────────────────────────────
        results: list[RetrievalResult] = []
        for final_score, nid in top_k:
            if nid not in self._graph:
                continue
            n = self._graph.V[nid]
            cos_direct = float(np.dot(q, n.v))
            self._reinforce(n)
            results.append(
                RetrievalResult(
                    node_id=nid,
                    score=final_score,
                    summary=n.summary,
                    ref_uri=n.ref_uri,
                    ref_type=n.ref_type,
                    tier=n.tier,
                    importance=n.importance,
                    urgency=n.urgency,
                    cos_direct=cos_direct,
                )
            )

        return results

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _reinforce(self, node: Node) -> None:
        """Apply one reinforcement step to a node (spacing effect).

        WHY effective strength → new s_base:
            We first compute the decayed effective strength at "now",
            then add delta_reinforce. This means a recently accessed node
            gets a smaller boost (already near max), while a decayed node
            gets a larger relative recovery — matching the spacing-effect
            psychology finding that spaced repetitions are more effective
            than massed practice.
        """
        now = time.time()
        s_eff = strength_effective(node, now, self._cfg)
        node.s_base = min(1.0, s_eff + self._cfg.delta_reinforce)
        node.t_accessed = now
        self._update_tier(node)

    def reinforce(self, node: Node) -> None:
        """Public alias for _reinforce."""
        self._reinforce(node)

    def _update_tier(self, node: Node) -> None:
        """Recompute and apply a node's tier based on current time.

        WHY: after reinforcement, a node's effective age drops, potentially
        moving it from LONG → MEDIUM or MEDIUM → SHORT. Updating the tier
        here keeps the graph and index consistent.
        """
        now = time.time()
        new_tier = tier_fn(node, now, self._cfg)
        if new_tier != node.tier:
            old_tier = node.tier
            self._graph.update_tier_set(node, old_tier, new_tier)
            self._index.move(node.id, node.v, old_tier, new_tier)
            node.tier = new_tier
            logger.debug(
                "_update_tier: %s %s → %s",
                node.id[:8], old_tier.name, new_tier.name,
            )

    def _decay_pass(self) -> None:
        """Materialise Ebbinghaus decay: s_eff → s_base, reset t_accessed.

        WHY materialise:
            The exponential decay is normally lazy (computed on read).
            Before a sleep phase we materialise it so that tier transitions
            are based on current (decayed) strength rather than peak
            strength. This prevents artificially high tier assignments for
            nodes that haven't been accessed in hours.
        """
        now = time.time()
        for node in list(self._graph.V.values()):
            s_eff = strength_effective(node, now, self._cfg)
            node.s_base = s_eff
            node.t_accessed = now
            self._update_tier(node)

    def _compute_centroid(
        self,
        nodes: list[Node],
        t: float,
        include_urgency: bool = False,
    ) -> np.ndarray:
        """Compute the weighted centroid of a list of nodes.

        Weights are the effective strength of each node at time t.
        When include_urgency=True, urgency_effective is added to the
        weight so that urgent nodes pull the centroid more strongly.
        The result is L2-normalised.

        WHY weighted centroid:
            Stronger (more reinforced) nodes should dominate the centroid
            direction. A uniform average would give equal weight to a
            half-forgotten node and a recently reinforced one.

        WHY include_urgency in sleep_promote:
            During the promote pass we want the evoked centroid to lean
            toward urgent memories that the user was focused on during the
            awake session. A task with high urgency should influence which
            long-term memories are pulled back to SHORT more than a
            low-urgency node of equal strength.

        Parameters
        ----------
        nodes          : list of Node objects to average.
        t              : current unix timestamp.
        include_urgency: if True, weight = s_eff + u_eff; else weight = s_eff.
        """
        if include_urgency:
            weights = np.array(
                [
                    strength_effective(n, t, self._cfg)
                    + urgency_effective(n, t, self._cfg)
                    for n in nodes
                ],
                dtype=np.float64,
            )
        else:
            weights = np.array(
                [strength_effective(n, t, self._cfg) for n in nodes],
                dtype=np.float64,
            )

        total = weights.sum()
        if total < 1e-12:
            weights = np.ones(len(nodes), dtype=np.float64)
            total = float(len(nodes))
        weights /= total

        centroid = np.zeros(self._cfg.d, dtype=np.float64)
        for w, n in zip(weights, nodes):
            centroid += w * n.v.astype(np.float64)

        norm = np.linalg.norm(centroid)
        if norm < 1e-12:
            # All nodes identical in vector space: return first node's vector.
            return nodes[0].v.copy()
        return (centroid / norm).astype(np.float32)

    # ── Status ─────────────────────────────────────────────────────────────────

    def status(self) -> dict:
        """Return a summary of current memory state.

        WHY: used by CLI and daemon to display human-readable stats
        without exposing internal graph structures.
        """
        return {
            "total_nodes": self._graph.total_nodes(),
            "total_edges": self._graph.total_edges(),
            "tier_counts": {
                "SHORT": self._graph.tier_size(Tier.SHORT),
                "MEDIUM": self._graph.tier_size(Tier.MEDIUM),
                "LONG": self._graph.tier_size(Tier.LONG),
            },
            "index_sizes": {
                "SHORT": self._index.size(Tier.SHORT),
                "MEDIUM": self._index.size(Tier.MEDIUM),
                "LONG": self._index.size(Tier.LONG),
            },
            "last_sleep_end": self._last_sleep_end,
        }
