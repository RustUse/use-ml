# use-ml-embedding

Embedding vector metadata primitives for `RustUse`.

## Experimental

`use-ml-embedding` is experimental while `use-ml` remains below `0.3.0`.

## Example

```rust
use use_ml_embedding::{EmbeddingDimension, EmbeddingModelName, EmbeddingVectorShape};

let model = EmbeddingModelName::new("text-embedding")?;
let dimension = EmbeddingDimension::new(384)?;
let shape = EmbeddingVectorShape::new(dimension);

assert_eq!(model.as_str(), "text-embedding");
assert_eq!(shape.dimension().get(), 384);
# Ok::<(), use_ml_embedding::EmbeddingError>(())
```

## Scope

- Embedding model names, vector IDs, dimensions, vector shapes, modalities, distance metrics, normalization labels, index labels, search labels, and vector formats.
- Metadata only; no embedding computation or vector search.

## Non-goals

- RAG pipelines, retrievers, chunkers, document stores, citations, context assembly, or reranking workflows.
- Vector search, embedding computation, model calls, or database clients.

## License

Licensed under either Apache-2.0 or MIT.
