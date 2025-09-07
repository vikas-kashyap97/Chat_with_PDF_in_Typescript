import "dotenv/config";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { RetrievalQAChain } from "langchain/chains";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

const embeddings = new OpenAIEmbeddings({
    model: "text-embedding-3-large"
});

const pdf = "./src/documents/Exploring_Forces.pdf";
const loader = new PDFLoader(pdf);
const docs = await loader.load();

//  Splitter fuction
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 20,
});

// Split the documents into chunks
const splitterDocs = await splitter.splitDocuments(docs);


const vectorStore = await MemoryVectorStore.fromDocuments(
  splitterDocs,
  embeddings
);



const vectorStoreRetriever = vectorStore.asRetriever();
const model = new ChatOpenAI({
  model: "gpt-3.5-turbo",
});



const chain = RetrievalQAChain.fromLLM(
  model,
  vectorStoreRetriever
);

const question = "If a ball is thrown upwards, it slows down, stops momentarily, and then falls back to the ground. Name the forces acting on the ball and specify their directions. (i) During its upward motion (ii) During its downward motion (iii) At its topmost position";
const res = await chain.call({ query: question });

console.log({ question: question, answer: res.text });
