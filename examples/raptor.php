<?php

declare(strict_types=1);

use NeuronAI\Chat\Messages\UserMessage;
use NeuronAI\Providers\Anthropic\Anthropic;
use NeuronAI\RAG\Embeddings\OpenAIEmbeddingsProvider;
use NeuronAI\RAG\RAG;
use NeuronAI\RAG\VectorStore\FileVectorStore;
use NeuronCore\RaptorRetrieval\RaptorRetrieval;

require_once __DIR__ . '/../vendor/autoload.php';

$llm = new Anthropic('ANTHROPIC_API_KEY', 'ANTHROPIC_MODEL');
$vectorStore = new FileVectorStore(__DIR__);
$embeddings = new OpenAIEmbeddingsProvider('OPENAI_API_KEY', 'OPENAI_MODEL');

$agent = RAG::make()
    ->withProvider($llm)
    ->setVectorStore($vectorStore)
    ->setEmbeddingsProvider($embeddings)
    ->setRetrieval(
        new RaptorRetrieval($vectorStore, $embeddings, $llm)
    );

$response = $agent->chat(new UserMessage('Hello'));

echo $response->getContent();
