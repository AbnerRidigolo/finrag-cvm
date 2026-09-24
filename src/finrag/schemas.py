"""Modelos de dados compartilhados entre ingestão, recuperação, agente e API."""

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    id: str
    text: str
    source: str
    article: str = ""
    position: int = 0
    context: str = Field(default="", description="Título do documento de origem")

    @property
    def search_text(self) -> str:
        """Texto indexado: o trecho precedido do título do documento e do artigo.

        Um artigo isolado ("A taxa de administração é de 1,2%...") não diz de qual fundo
        está falando. Prefixar o título devolve esse contexto aos dois retrievers.
        """
        header = " | ".join(p for p in (self.context, self.article) if p)
        return f"{header}\n{self.text}" if header else self.text


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float
    retriever: str = Field(description="dense, sparse, hybrid ou rerank")


class Source(BaseModel):
    id: str
    source: str
    title: str = ""
    article: str
    excerpt: str
    score: float
    text: str = Field(default="", exclude=True, description="Texto completo, usado pelo juiz")


class Usage(BaseModel):
    latency_ms: float = 0.0
    stage_latency_ms: dict[str, float] = {}
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    llm_calls: int = 0


class Answer(BaseModel):
    question: str
    answer: str
    route: str
    sources: list[Source] = []
    context: str = Field(default="", exclude=True, description="Contexto enviado ao LLM")
    usage: Usage = Usage()
