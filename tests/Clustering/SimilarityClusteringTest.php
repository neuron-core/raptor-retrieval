<?php

declare(strict_types=1);

namespace NeuronCore\RaptorRetrieval\Tests\Clustering;

use NeuronCore\RaptorRetrieval\Clustering\SimilarityClustering;
use NeuronCore\RaptorRetrieval\TreeNode;
use PHPUnit\Framework\TestCase;

class SimilarityClusteringTest extends TestCase
{
    public function testClusterEmptyArray(): void
    {
        $clustering = new SimilarityClustering();
        $result = $clustering->cluster([]);

        $this->assertEmpty($result);
    }

    public function testClusterSingleNode(): void
    {
        $clustering = new SimilarityClustering();

        $node = new TreeNode();
        $node->id = 'node1';
        $node->content = 'Test content';
        $node->embedding = [1.0, 0.0, 0.0];

        $result = $clustering->cluster([$node]);

        $this->assertCount(1, $result);
        $this->assertCount(1, $result[0]);
        $this->assertSame($node, $result[0][0]);
    }

    public function testClusterSimilarNodes(): void
    {
        $clustering = new SimilarityClustering(0.8, 5); // High similarity threshold

        $node1 = new TreeNode();
        $node1->id = 'node1';
        $node1->content = 'Similar content 1';
        $node1->embedding = [1.0, 0.0, 0.0];

        $node2 = new TreeNode();
        $node2->id = 'node2';
        $node2->content = 'Similar content 2';
        $node2->embedding = [0.9, 0.1, 0.0]; // Very similar to node1

        $result = $clustering->cluster([$node1, $node2]);

        $this->assertCount(1, $result); // Should be clustered together
        $this->assertCount(2, $result[0]);
    }

    public function testClusterDissimilarNodes(): void
    {
        $clustering = new SimilarityClustering(0.8, 5); // High similarity threshold

        $node1 = new TreeNode();
        $node1->id = 'node1';
        $node1->content = 'Content about cats';
        $node1->embedding = [1.0, 0.0, 0.0];

        $node2 = new TreeNode();
        $node2->id = 'node2';
        $node2->content = 'Content about dogs';
        $node2->embedding = [0.0, 1.0, 0.0]; // Orthogonal to node1

        $result = $clustering->cluster([$node1, $node2]);

        $this->assertCount(2, $result); // Should be in separate clusters
        $this->assertCount(1, $result[0]);
        $this->assertCount(1, $result[1]);
    }

    public function testClusterMaxSize(): void
    {
        $clustering = new SimilarityClustering(0.1, 2); // Low threshold, max 2 per cluster

        $nodes = [];
        for ($i = 0; $i < 5; $i++) {
            $node = new TreeNode();
            $node->id = "node{$i}";
            $node->content = "Content {$i}";
            $node->embedding = [1.0, 0.0, 0.0]; // All very similar
            $nodes[] = $node;
        }

        $result = $clustering->cluster($nodes);

        // With max cluster size of 2, we should have at least 3 clusters for 5 nodes
        $this->assertGreaterThanOrEqual(3, \count($result));

        // Each cluster should have at most 2 nodes
        foreach ($result as $cluster) {
            $this->assertLessThanOrEqual(2, \count($cluster));
        }
    }

    public function testClusterWithIncompatibleEmbeddings(): void
    {
        $clustering = new SimilarityClustering();

        $node1 = new TreeNode();
        $node1->id = 'node1';
        $node1->content = 'Content 1';
        $node1->embedding = [1.0, 0.0];

        $node2 = new TreeNode();
        $node2->id = 'node2';
        $node2->content = 'Content 2';
        $node2->embedding = [1.0, 0.0, 0.0]; // Different dimension

        $result = $clustering->cluster([$node1, $node2]);

        // Should handle gracefully and put in separate clusters
        $this->assertCount(2, $result);
    }
}
