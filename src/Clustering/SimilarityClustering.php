<?php

declare(strict_types=1);

namespace NeuronAI\Raptor\Clustering;

use NeuronAI\RAG\VectorSimilarity;
use NeuronAI\Raptor\TreeNode;

class SimilarityClustering implements ClusteringInterface
{
    public function __construct(
        private readonly float $similarityThreshold = 0.7,
        private readonly int $maxClusterSize = 8
    ) {
    }

    /**
     * @param TreeNode[] $nodes
     * @return array<TreeNode[]> Array of clusters
     */
    public function cluster(array $nodes): array
    {
        $clusters = [];
        $used = \array_fill(0, \count($nodes), false);
        $counter = \count($nodes);

        for ($i = 0; $i < $counter; $i++) {
            if ($used[$i]) {
                continue;
            }

            $cluster = [$nodes[$i]];
            $used[$i] = true;

            // Find similar nodes for this cluster
            for ($j = $i + 1; $j < \count($nodes); $j++) {
                if ($used[$j]) {
                    continue;
                }

                try {
                    $similarity = VectorSimilarity::cosineSimilarity(
                        $nodes[$i]->embedding,
                        $nodes[$j]->embedding
                    );

                    // If similar enough and cluster not too big, add to cluster
                    if ($similarity > $this->similarityThreshold && \count($cluster) < $this->maxClusterSize) {
                        $cluster[] = $nodes[$j];
                        $used[$j] = true;
                    }
                } catch (\Exception) {
                    // Skip this comparison if vectors are incompatible
                    continue;
                }
            }

            $clusters[] = $cluster;
        }

        return $clusters;
    }
}
