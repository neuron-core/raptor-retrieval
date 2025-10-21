<?php

declare(strict_types=1);

namespace NeuronCore\RaptorRetrieval\Tests;

use NeuronAI\Chat\Messages\AssistantMessage;
use NeuronAI\Chat\Messages\UserMessage;
use NeuronAI\Providers\AIProviderInterface;
use NeuronAI\RAG\Document;
use NeuronAI\RAG\Embeddings\EmbeddingsProviderInterface;
use NeuronAI\RAG\VectorStore\VectorStoreInterface;
use NeuronCore\RaptorRetrieval\Clustering\ClusteringInterface;
use NeuronCore\RaptorRetrieval\RaptorRetrieval;
use PHPUnit\Framework\MockObject\MockObject;
use PHPUnit\Framework\TestCase;

class RaptorRetrievalTest extends TestCase
{
    private VectorStoreInterface&MockObject $vectorStore;
    private EmbeddingsProviderInterface&MockObject $embeddingProvider;
    private AIProviderInterface&MockObject $summarizationProvider;
    private ClusteringInterface&MockObject $clustering;
    private RaptorRetrieval $raptorRetrieval;

    protected function setUp(): void
    {
        $this->vectorStore = $this->createMock(VectorStoreInterface::class);
        $this->embeddingProvider = $this->createMock(EmbeddingsProviderInterface::class);
        $this->summarizationProvider = $this->createMock(AIProviderInterface::class);
        $this->clustering = $this->createMock(ClusteringInterface::class);

        $this->raptorRetrieval = new RaptorRetrieval(
            $this->vectorStore,
            $this->embeddingProvider,
            $this->summarizationProvider,
            $this->clustering
        );
    }

    public function testRetrieveWithEmptyResults(): void
    {
        $query = new UserMessage('test query');
        $queryEmbedding = [0.1, 0.2, 0.3];

        $this->embeddingProvider
            ->expects($this->once())
            ->method('embedText')
            ->with('test query')
            ->willReturn($queryEmbedding);

        $this->vectorStore
            ->expects($this->once())
            ->method('similaritySearch')
            ->with($queryEmbedding)
            ->willReturn([]);

        $result = $this->raptorRetrieval->retrieve($query);

        $this->assertEmpty($result);
    }

    public function testRetrieveWithSingleDocument(): void
    {
        $query = new UserMessage('test query');
        $queryEmbedding = [0.1, 0.2, 0.3];

        $document = new Document('Test document content');
        $document->id = 'doc-1';
        $document->embedding = [0.5, 0.6, 0.7];

        $this->embeddingProvider
            ->expects($this->once())
            ->method('embedText')
            ->with('test query')
            ->willReturn($queryEmbedding);

        $this->vectorStore
            ->expects($this->once())
            ->method('similaritySearch')
            ->with($queryEmbedding)
            ->willReturn([$document]);

        // With single document, clustering is not called due to base case
        $this->clustering
            ->expects($this->never())
            ->method('cluster');

        $result = $this->raptorRetrieval->retrieve($query);

        $this->assertCount(1, $result);
        $this->assertInstanceOf(Document::class, $result[0]);
        $this->assertEquals('Test document content', $result[0]->getContent());
    }

    public function testRetrieveWithMultipleDocuments(): void
    {
        $query = new UserMessage('test query');
        $queryEmbedding = [0.1, 0.2, 0.3];

        $doc1 = new Document('Document 1 content');
        $doc1->id = 'doc-1';
        $doc1->embedding = [0.5, 0.6, 0.7];

        $doc2 = new Document('Document 2 content');
        $doc2->id = 'doc-2';
        $doc2->embedding = [0.8, 0.9, 1.0];

        $this->embeddingProvider
            ->expects($this->exactly(2))
            ->method('embedText')
            ->willReturnMap([
                ['test query', $queryEmbedding],
                ['Test summary', [0.2, 0.3, 0.4]], // For summary generation
            ]);

        $this->vectorStore
            ->expects($this->once())
            ->method('similaritySearch')
            ->with($queryEmbedding)
            ->willReturn([$doc1, $doc2]);

        // Mock clustering to create clusters that will trigger summarization
        $this->clustering
            ->expects($this->once())
            ->method('cluster')
            ->willReturnCallback(fn (array $nodes): array =>
                // Return a cluster with multiple nodes to trigger summarization
                [[$nodes[0], $nodes[1]]]);

        // Mock AI provider for summarization
        $summaryResponse = new AssistantMessage('Test summary');
        $this->summarizationProvider
            ->expects($this->once())
            ->method('chat')
            ->willReturn($summaryResponse);

        $result = $this->raptorRetrieval->retrieve($query);

        $this->assertNotEmpty($result);
        $this->assertContainsOnlyInstancesOf(Document::class, $result);
    }

    public function testRetrieveHandlesIncompatibleEmbeddings(): void
    {
        $query = new UserMessage('test query');
        $queryEmbedding = [0.1, 0.2, 0.3];

        $document = new Document('Test document content');
        $document->id = 'doc-1';
        $document->embedding = [0.5, 0.6]; // Different dimension - will cause exception in similarity

        $this->embeddingProvider
            ->expects($this->once())
            ->method('embedText')
            ->with('test query')
            ->willReturn($queryEmbedding);

        $this->vectorStore
            ->expects($this->once())
            ->method('similaritySearch')
            ->with($queryEmbedding)
            ->willReturn([$document]);

        // With single document, clustering is not called due to base case
        $this->clustering
            ->expects($this->never())
            ->method('cluster');

        // Should not throw exception, should handle gracefully
        // Documents with incompatible embeddings are filtered out
        $result = $this->raptorRetrieval->retrieve($query);

        $this->assertEmpty($result); // Incompatible embeddings get filtered out
    }

    public function testConstructorWithDefaultClustering(): void
    {
        $retrieval = new RaptorRetrieval(
            $this->vectorStore,
            $this->embeddingProvider,
            $this->summarizationProvider
        );

        $this->assertInstanceOf(RaptorRetrieval::class, $retrieval);
    }

    public function testRetrieveWithTwoDocumentsClustering(): void
    {
        $query = new UserMessage('test query');
        $queryEmbedding = [0.1, 0.2, 0.3];

        $doc1 = new Document('Document 1 content');
        $doc1->id = 'doc-1';
        $doc1->embedding = [0.5, 0.6, 0.7];

        $doc2 = new Document('Document 2 content');
        $doc2->id = 'doc-2';
        $doc2->embedding = [0.8, 0.9, 1.0];

        $this->embeddingProvider
            ->method('embedText')
            ->willReturnMap([
                ['test query', $queryEmbedding],
                ['Test summary', [0.2, 0.3, 0.4]],
            ]);

        $this->vectorStore
            ->expects($this->once())
            ->method('similaritySearch')
            ->willReturn([$doc1, $doc2]);

        // Clustering will group both documents together
        $this->clustering
            ->expects($this->once())
            ->method('cluster')
            ->willReturnCallback(function (array $nodes): array {
                return [[$nodes[0], $nodes[1]]]; // Single cluster with both nodes
            });

        $summaryResponse = new AssistantMessage('Test summary');
        $this->summarizationProvider
            ->expects($this->once())
            ->method('chat')
            ->willReturn($summaryResponse);

        $result = $this->raptorRetrieval->retrieve($query);

        $this->assertNotEmpty($result);
        $this->assertContainsOnlyInstancesOf(Document::class, $result);
        // Should have original docs plus summary (flattened tree)
        $this->assertCount(3, $result);
    }
}
