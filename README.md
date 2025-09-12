# Recursive Abstractive Processing for Tree-Organized Retrieval

This module implements the RAPTOR retreival strategy for the Neuron PHP AI framework.

## What is Neuron?

Neuron is a PHP framework for creating and orchestrating AI Agents. It allows you to integrate AI entities in your existing
PHP applications with a powerful and flexible architecture. We provide tools for the entire agentic application development lifecycle,
from LLM interfaces, to data loading, to multi-agent orchestration, to monitoring and debugging.
In addition, we provide tutorials and other educational content to help you get started using AI Agents in your projects.

**[Go to the official documentation](https://neuron.inspector.dev/)**

[**Video Tutorial**](https://www.youtube.com/watch?v=oSA1bP_j41w)

[![Neuron & Inspector](./docs/youtube.png)](https://www.youtube.com/watch?v=oSA1bP_j41w)

---

## Requirements

- PHP: ^8.1
- Neuron: ^2.0

## Install

Install the latest version of the package:

```
composer require neuron-core/neuron-raptor-retreival
```

## How to use in your agent

Or use the RAPTOR component directly into the agent. RAPTOR needs a vector store, an embeddings provider and uses an LLM
to perform the summarization:

```php
use NeuronAI\RAG\Retrieval\RetrievalInterface;
use NeuronAI\Raptor\RaptorRetrieval;

class WorkoutTipsAgent extends RAG
{
    protected function retrieval(): RetrievalInterface
    {
        return new RaptorRetrieval(
            $this->resolveVectorStore(),
            $this->resolveEmbeddingsProvider(),
            $this->resolveProvider(), // Used for summarization
        );
    }

    protected function embeddings(): EmbeddingsProviderInterface
    {
        return new ...
    }

    protected function vectorStore(): VectorStoreInterface
    {
        return new ...
    }
}
```

## Clustering strategy

RAPTOR algorithm uses a clustering strategy to group the retrieved documents into clusters. There two common clustering strategies
you can use based on your scenario:

### Similarity Clustering (default)

This strategy groups the retrieved documents based on their similarity.

```php
use NeuronAI\RAG\Retrieval\RetrievalInterface;
use NeuronAI\Raptor\RaptorRetrieval;
use NeuronAI\Raptor\Clustering\SimilarityClusteringStrategy;

class WorkoutTipsAgent extends RAG
{
    protected function retrieval(): RetrievalInterface
    {
        return new RaptorRetrieval(
            $this->resolveVectorStore(),
            $this->resolveEmbeddingsProvider(),
            $this->resolveProvider(), // Used for summarization
            new SimilarityClusteringStrategy()
        );
    }

    protected function embeddings(): EmbeddingsProviderInterface
    {
        return new ...
    }

    protected function vectorStore(): VectorStoreInterface
    {
        return new ...
    }
}
```

### Gaussian Mixture Clustering

Unlike hard clustering methods such as K-Means, which assign each point to a single cluster based on the closest centroid,
GMM performs soft clustering by assigning each point a probability of belonging to multiple clusters.

```php
use NeuronAI\RAG\Retrieval\RetrievalInterface;
use NeuronAI\Raptor\RaptorRetrieval;
use NeuronAI\Raptor\Clustering\GaussianMixtureClustering;

class WorkoutTipsAgent extends RAG
{
    protected function retrieval(): RetrievalInterface
    {
        return new RaptorRetrieval(
            $this->resolveVectorStore(),
            $this->resolveEmbeddingsProvider(),
            $this->resolveProvider(), // Used for summarization
            new GaussianMixtureClustering()
        );
    }

    protected function embeddings(): EmbeddingsProviderInterface
    {
        return new ...
    }

    protected function vectorStore(): VectorStoreInterface
    {
        return new ...
    }
}
```
