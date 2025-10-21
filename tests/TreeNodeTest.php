<?php

declare(strict_types=1);

namespace NeuronAI\Raptor\Tests;

use NeuronAI\RAG\Document;
use NeuronCore\RaptorRetrieval\TreeNode;
use PHPUnit\Framework\TestCase;

class TreeNodeTest extends TestCase
{
    public function testTreeNodeCreation(): void
    {
        $node = new TreeNode();
        $node->id = 'test-id';
        $node->content = 'Test content';
        $node->embedding = [0.1, 0.2, 0.3];
        $node->level = 1;

        $this->assertEmpty($node->childNodes);
        $this->assertNull($node->parentNode);
        $this->assertNull($node->originalDocument);
    }

    public function testTreeNodeWithChildren(): void
    {
        $parentNode = new TreeNode();
        $parentNode->id = 'parent';
        $parentNode->content = 'Parent content';

        $childNode1 = new TreeNode();
        $childNode1->id = 'child1';
        $childNode1->content = 'Child 1 content';
        $childNode1->parentNode = $parentNode;

        $childNode2 = new TreeNode();
        $childNode2->id = 'child2';
        $childNode2->content = 'Child 2 content';
        $childNode2->parentNode = $parentNode;

        $parentNode->childNodes = [$childNode1, $childNode2];

        $this->assertCount(2, $parentNode->childNodes);
        $this->assertSame($parentNode, $childNode1->parentNode);
        $this->assertSame($parentNode, $childNode2->parentNode);
    }

    public function testTreeNodeWithOriginalDocument(): void
    {
        $document = new Document('Original content');
        $document->id = 'doc-123';

        $node = new TreeNode();
        $node->id = 'node-123';
        $node->content = 'Original content';
        $node->originalDocument = $document;

        $this->assertSame($document, $node->originalDocument);
        $this->assertEquals('doc-123', $node->originalDocument->getId());
        $this->assertEquals('Original content', $node->originalDocument->getContent());
    }
}
