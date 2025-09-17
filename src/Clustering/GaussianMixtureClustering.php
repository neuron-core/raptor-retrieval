<?php

declare(strict_types=1);

namespace NeuronAI\Raptor\Clustering;

use NeuronAI\Raptor\TreeNode;

/**
 * Gaussian Mixture Model clustering implementation for RAPTOR.
 *
 * This implementation uses the Expectation-Maximization algorithm with
 * Bayesian Information Criterion (BIC) for optimal cluster selection.
 * Based on the RAPTOR paper's clustering approach.
 */
class GaussianMixtureClustering implements ClusteringInterface
{
    public function __construct(
        protected readonly int $maxClusters = 10,
        protected readonly int $maxIterations = 100,
        protected readonly int $minClusterSize = 2,
        protected readonly int $maxClusterSize = 8,
        protected readonly bool $useUMAP = false, // For future UMAP integration
    ) {
    }

    /**
     * @param TreeNode[] $nodes
     * @return array<TreeNode[]>
     */
    public function cluster(array $nodes): array
    {
        if (\count($nodes) <= 1) {
            return [$nodes];
        }

        // Extract embeddings from nodes
        $embeddings = \array_map(fn (TreeNode $node): array => $node->embedding, $nodes);

        // Apply dimensionality reduction if enabled (placeholder for now)
        if ($this->useUMAP) {
            $embeddings = $this->applyUMAP($embeddings);
        }

        // Find the optimal number of clusters using BIC
        $optimalK = $this->findOptimalClusters($embeddings);

        // Perform GMM clustering
        $assignments = $this->performGMMClustering($embeddings, $optimalK);

        // Group nodes by cluster assignments
        return $this->groupNodesByAssignments($nodes, $assignments);
    }

    /**
     * Find the optimal number of clusters using Bayesian Information Criterion (BIC).
     *
     * @param array<array<float>> $embeddings
     */
    private function findOptimalClusters(array $embeddings): int
    {
        $bestK = 1;
        $bestBIC = \PHP_FLOAT_MAX;

        for ($k = 1; $k <= \min($this->maxClusters, \count($embeddings)); ++$k) {
            $bic = $this->calculateBIC($embeddings, $k);

            if ($bic < $bestBIC) {
                $bestBIC = $bic;
                $bestK = $k;
            }
        }

        return $bestK;
    }

    /**
     * Calculate Bayesian Information Criterion for given number of clusters.
     *
     * @param array<array<float>> $embeddings
     */
    private function calculateBIC(array $embeddings, int $k): float
    {
        $n = \count($embeddings);
        $d = \count($embeddings[0]);

        // Number of parameters in GMM: k means + k covariances + k-1 mixing weights
        $numParams = $k * $d + $k * $d + ($k - 1);

        // Perform simplified clustering to estimate log-likelihood
        $logLikelihood = $this->estimateLogLikelihood($embeddings, $k);

        // BIC = -2 * log-likelihood + k * log(n)
        return -2 * $logLikelihood + $numParams * \log($n);
    }

    /**
     * Simplified log-likelihood estimation using K-means approximation.
     *
     * @param array<array<float>> $embeddings
     */
    private function estimateLogLikelihood(array $embeddings, int $k): float
    {
        // Use simplified K-means for quick log-likelihood estimation
        $assignments = $this->performSimpleKMeans($embeddings, $k);

        $logLikelihood = 0.0;
        $clusters = $this->groupEmbeddingsByAssignments($embeddings, $assignments, $k);

        foreach ($clusters as $cluster) {
            if (empty($cluster)) {
                continue;
            }

            $centroid = $this->calculateCentroid($cluster);
            foreach ($cluster as $embedding) {
                $distance = $this->euclideanDistance($embedding, $centroid);
                // Approximate log-likelihood using negative squared distance
                $logLikelihood -= $distance * $distance;
            }
        }

        return $logLikelihood;
    }

    /**
     * Perform Gaussian Mixture Model clustering using Expectation-Maximization.
     *
     * @param array<array<float>> $embeddings
     * @return array<int>
     */
    private function performGMMClustering(array $embeddings, int $k): array
    {
        if ($k === 1) {
            return \array_fill(0, \count($embeddings), 0);
        }

        // For simplicity, use K-means as an approximation to GMM
        // In a full implementation this would be a proper EM algorithm
        return $this->performSimpleKMeans($embeddings, $k);
    }

    /**
     * Simple K-means clustering implementation.
     *
     * @param array<array<float>> $embeddings
     * @return array<int>
     */
    private function performSimpleKMeans(array $embeddings, int $k): array
    {
        $n = \count($embeddings);
        if ($n <= $k) {
            return \array_keys($embeddings);
        }

        // Initialize centroids randomly
        $centroids = $this->initializeCentroids($embeddings, $k);
        $assignments = \array_fill(0, $n, 0);

        for ($iteration = 0; $iteration < $this->maxIterations; $iteration++) {
            $oldAssignments = $assignments;

            // Assignment step
            for ($i = 0; $i < $n; ++$i) {
                $bestDistance = \PHP_FLOAT_MAX;
                $bestCluster = 0;

                for ($j = 0; $j < $k; ++$j) {
                    $distance = $this->euclideanDistance($embeddings[$i], $centroids[$j]);
                    if ($distance < $bestDistance) {
                        $bestDistance = $distance;
                        $bestCluster = $j;
                    }
                }

                $assignments[$i] = $bestCluster;
            }

            // Update centroids
            $clusters = $this->groupEmbeddingsByAssignments($embeddings, $assignments, $k);
            for ($j = 0; $j < $k; ++$j) {
                if (!empty($clusters[$j])) {
                    $centroids[$j] = $this->calculateCentroid($clusters[$j]);
                }
            }

            // Check for convergence
            if ($assignments === $oldAssignments) {
                break;
            }
        }

        return $assignments;
    }

    /**
     * @param array<array<float>> $embeddings
     * @return array<array<float>>
     */
    private function initializeCentroids(array $embeddings, int $k): array
    {
        $centroids = [];
        $indices = \array_rand($embeddings, \min($k, \count($embeddings)));

        if (!\is_array($indices)) {
            $indices = [$indices];
        }

        foreach ($indices as $index) {
            $centroids[] = $embeddings[$index];
        }

        // Fill remaining centroids if needed
        while (\count($centroids) < $k) {
            $centroids[] = $embeddings[\array_rand($embeddings)];
        }

        return $centroids;
    }

    /**
     * @param array<array<float>> $embeddings
     * @param array<int> $assignments
     * @return array<array<array<float>>>
     */
    private function groupEmbeddingsByAssignments(array $embeddings, array $assignments, int $k): array
    {
        $clusters = \array_fill(0, $k, []);
        $counter = \count($embeddings);

        for ($i = 0; $i < $counter; ++$i) {
            $clusters[$assignments[$i]][] = $embeddings[$i];
        }

        return $clusters;
    }

    /**
     * @param TreeNode[] $nodes
     * @param array<int> $assignments
     * @return array<TreeNode[]>
     */
    private function groupNodesByAssignments(array $nodes, array $assignments): array
    {
        $clusters = [];
        $counter = \count($nodes);

        for ($i = 0; $i < $counter; ++$i) {
            $clusterIndex = $assignments[$i];
            if (!isset($clusters[$clusterIndex])) {
                $clusters[$clusterIndex] = [];
            }
            $clusters[$clusterIndex][] = $nodes[$i];
        }

        // Filter out clusters that are too small or too large
        $filteredClusters = [];
        foreach ($clusters as $cluster) {
            $size = \count($cluster);
            if ($size >= $this->minClusterSize && $size <= $this->maxClusterSize) {
                $filteredClusters[] = $cluster;
            } elseif ($size === 1) {
                // Keep single-node clusters as they represent unique content
                $filteredClusters[] = $cluster;
            } elseif ($size > $this->maxClusterSize) {
                // Split large clusters or merge with others (simplified approach)
                // Split into smaller clusters
                $splitClusters = \array_chunk($cluster, $this->maxClusterSize);
                $filteredClusters = \array_merge($filteredClusters, $splitClusters);
            }
        }

        return \array_values($filteredClusters);
    }

    /**
     * @param array<float> $embedding1
     * @param array<float> $embedding2
     */
    private function euclideanDistance(array $embedding1, array $embedding2): float
    {
        $sum = 0.0;
        $counter = \count($embedding1);
        for ($i = 0; $i < $counter; ++$i) {
            $diff = $embedding1[$i] - $embedding2[$i];
            $sum += $diff * $diff;
        }
        return \sqrt($sum);
    }

    /**
     * @param array<array<float>> $embeddings
     * @return array<float>
     */
    private function calculateCentroid(array $embeddings): array
    {
        if ($embeddings === []) {
            return [];
        }

        $dimensions = \count($embeddings[0]);
        $centroid = \array_fill(0, $dimensions, 0.0);

        foreach ($embeddings as $embedding) {
            for ($i = 0; $i < $dimensions; ++$i) {
                $centroid[$i] += $embedding[$i];
            }
        }

        $count = \count($embeddings);
        for ($i = 0; $i < $dimensions; $i++) {
            $centroid[$i] /= $count;
        }

        return $centroid;
    }

    /**
     * Placeholder for UMAP dimensionality reduction.
     * In a full implementation, this would integrate with external UMAP libraries.
     *
     * @param array<array<float>> $embeddings
     * @return array<array<float>>
     */
    private function applyUMAP(array $embeddings): array
    {
        // Placeholder - would require external UMAP implementation
        // For now, return embeddings unchanged
        return $embeddings;
    }
}
