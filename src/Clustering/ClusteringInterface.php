<?php

declare(strict_types=1);

namespace NeuronCore\RaptorRetrieval\Clustering;

use NeuronCore\RaptorRetrieval\TreeNode;

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
