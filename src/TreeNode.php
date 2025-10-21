<?php

declare(strict_types=1);

namespace NeuronCore\RaptorRetrieval;

use NeuronAI\RAG\Document;

class TreeNode
{
    public string $id;

    public string $content;

    /**
     * @var array<float>
     */
    public array $embedding = [];

    public int $level = 0;

    /**
     * @var TreeNode[]
     */
    public array $childNodes = [];

    public ?TreeNode $parentNode = null;

    public ?Document $originalDocument = null;
}
