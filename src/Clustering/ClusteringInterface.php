<?php

declare(strict_types=1);

namespace NeuronAI\Raptor\Clustering;

use NeuronAI\Raptor\TreeNode;

interface ClusteringInterface
{
    /**
     * Cluster nodes based on similarity.
     *
     * @param TreeNode[] $nodes
     * @return array<TreeNode[]> Array of clusters
     */
    public function cluster(array $nodes): array;
}
