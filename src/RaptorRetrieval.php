<?php

declare(strict_types=1);

namespace NeuronAI\Raptor;

use NeuronAI\Chat\Messages\Message;
use NeuronAI\Chat\Messages\UserMessage;
use NeuronAI\Providers\AIProviderInterface;
use NeuronAI\RAG\Document;
use NeuronAI\RAG\Embeddings\EmbeddingsProviderInterface;
use NeuronAI\Raptor\Clustering\ClusteringInterface;
use NeuronAI\Raptor\Clustering\SimilarityClustering;
use NeuronAI\RAG\Retrieval\RetrievalInterface;
use NeuronAI\RAG\VectorSimilarity;
use NeuronAI\RAG\VectorStore\VectorStoreInterface;

/**
 * RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval)
 *
 * This retrieval strategy builds a hierarchical tree structure from candidate documents
 * by clustering similar chunks and creating summarized parent nodes. During retrieval,
 * it can access information at different levels of abstraction, from specific details
 * to high-level themes.
 *
 * Usage:
 * ```php
 * $raptorRetrieval = new RaptorRetrieval(
 *     $vectorStore,
 *     $embeddingsProvider,
 *     $aiProvider, // Used for summarization
 *     new SimilarityClustering(0.7, 8) // Clustering strategy
 * );
 *
 * $rag = RAG::make()
 *     ->withVectorStore($vectorStore)
 *     ->withEmbeddingsProvider($embeddingsProvider)
 *     ->withProvider($aiProvider)
 *     ->setRetrieval($raptorRetrieval);
 * ```
 */
class RaptorRetrieval implements RetrievalInterface
{
    public function __construct(
        private readonly VectorStoreInterface $vectorStore,
        private readonly EmbeddingsProviderInterface $embeddingProvider,
        private readonly AIProviderInterface $summarizationProvider,
        private readonly ClusteringInterface $clustering = new SimilarityClustering(),
    ) {
    }

    /**
     * @return Document[]
     */
    public function retrieve(Message $query): array
    {
        $queryText = $query->getContent();
        $queryEmbedding = $this->embeddingProvider->embedText($queryText);

        // Step 1: Get candidate documents using similarity search
        $candidateDocuments = $this->vectorStore->similaritySearch($queryEmbedding);

        if (empty($candidateDocuments)) {
            return [];
        }

        // Step 2: Build RAPTOR tree from candidates
        $tree = $this->buildTree($candidateDocuments);

        // Step 3: Retrieve from the tree using the collapsed tree method
        return $this->collapsedTreeRetrieval($tree, $queryEmbedding);
    }

    /**
     * @param Document[] $documents
     * @return TreeNode[]
     */
    private function buildTree(array $documents): array
    {
        // Convert documents to tree nodes
        $currentLevel = [];
        foreach ($documents as $document) {
            $node = new TreeNode();
            $node->id = (string) $document->getId();
            $node->content = $document->getContent();
            $node->embedding = $document->getEmbedding();
            $node->level = 0;
            $node->originalDocument = $document;
            $currentLevel[] = $node;
        }

        // Build tree recursively
        return $this->buildTreeRecursively($currentLevel);
    }

    /**
     * @param TreeNode[] $nodes
     * @return TreeNode[]
     */
    private function buildTreeRecursively(array $nodes): array
    {
        if (\count($nodes) <= 1) {
            return $nodes; // Base case: root reached
        }

        $clusters = $this->clustering->cluster($nodes);
        $nextLevel = [];

        foreach ($clusters as $cluster) {
            if (\count($cluster) === 1) {
                // Single node - promote to the next level
                $nextLevel[] = $cluster[0];
                continue;
            }

            // Create summary node for cluster
            $summaryNode = $this->createSummaryNode($cluster);
            $nextLevel[] = $summaryNode;
        }

        return $this->buildTreeRecursively($nextLevel);
    }

    /**
     * @param TreeNode[] $cluster
     */
    private function createSummaryNode(array $cluster): TreeNode
    {
        // Combine content from cluster nodes
        $combinedContent = '';
        foreach ($cluster as $node) {
            $combinedContent .= $node->content."\n\n";
        }

        // Generate summary
        $summary = $this->generateSummary(\trim($combinedContent));

        // Create parent node
        $parentNode = new TreeNode();
        $parentNode->id = 'summary_'.\uniqid();
        $parentNode->content = $summary;
        $parentNode->embedding = $this->embeddingProvider->embedText($summary);
        $parentNode->level = $cluster[0]->level + 1;
        $parentNode->childNodes = $cluster;

        // Set parent references
        foreach ($cluster as $childNode) {
            $childNode->parentNode = $parentNode;
        }

        return $parentNode;
    }

    private function generateSummary(string $content): string
    {
        $response = $this->summarizationProvider->chat([
            new UserMessage("Summarize the following text, capturing the key information and themes:\n\n{$content}"),
        ]);

        return $response->getContent();
    }

    /**
     * @param TreeNode[] $tree
     * @param array<float> $queryEmbedding
     * @return Document[]
     */
    private function collapsedTreeRetrieval(array $tree, array $queryEmbedding): array
    {
        // Flatten the tree to get all nodes
        $allNodes = $this->flattenTree($tree);

        // Score all nodes by similarity to the query
        $scoredNodes = [];
        foreach ($allNodes as $node) {
            try {
                $scoredNodes[] = [
                    'node' => $node,
                    'score' => VectorSimilarity::cosineSimilarity($queryEmbedding, $node->embedding)
                ];
            } catch (\Exception) {
                // Skip nodes with incompatible embeddings
                continue;
            }
        }

        // Sort by similarity score
        \usort($scoredNodes, fn (array $a, array $b): int => $b['score'] <=> $a['score']);

        // Convert nodes to documents
        return \array_map(fn (array $scoredNode): Document => $this->convertNodeToDocument($scoredNode['node']), $scoredNodes);
    }

    /**
     * @param TreeNode[] $tree
     * @return TreeNode[]
     */
    private function flattenTree(array $tree): array
    {
        $allNodes = [];

        foreach ($tree as $node) {
            $allNodes[] = $node;
            $allNodes = \array_merge($allNodes, $this->flattenNode($node));
        }

        return $allNodes;
    }

    /**
     * @return TreeNode[]
     */
    private function flattenNode(TreeNode $node): array
    {
        $nodes = [];

        foreach ($node->childNodes as $child) {
            $nodes[] = $child;
            $nodes = \array_merge($nodes, $this->flattenNode($child));
        }

        return $nodes;
    }


    private function convertNodeToDocument(TreeNode $node): Document
    {
        // If it's an original document, return it
        if ($node->originalDocument instanceof \NeuronAI\RAG\Document) {
            return $node->originalDocument;
        }

        // Create document from summary node
        $document = new Document($node->content);
        $document->id = $node->id;
        $document->embedding = $node->embedding;
        $document->addMetadata('raptor_level', $node->level);
        $document->addMetadata('raptor_type', 'summary');

        return $document;
    }

}
