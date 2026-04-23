// Background service worker for μDeep Blocker extension
import { pipeline, env } from '@xenova/transformers';
import * as ort from 'onnxruntime-web';

// Browser detection and API abstraction
declare const chrome: any;
declare const browser: any;

const isFirefox = typeof browser !== 'undefined';
const browserAPI = isFirefox ? browser : chrome;

// Configuration
const BLOCK_THRESHOLD = 0.65;
const MODEL_PATH = browserAPI.runtime.getURL('models/udeep_classifier.onnx');

// Global variables for AI models
let embeddingPipeline: any = null;
let classifierSession: ort.InferenceSession | null = null;
let isModelsLoaded = false;

// Statistics tracking
let blockedRequests = 0;
let totalRequests = 0;

// Browser-agnostic storage
async function getStorageData(keys: string[]): Promise<any> {
  if (isFirefox) {
    return browserAPI.storage.local.get(keys);
  } else {
    return new Promise((resolve) => {
      chrome.storage.local.get(keys, resolve);
    });
  }
}

async function setStorageData(data: any): Promise<void> {
  if (isFirefox) {
    await browserAPI.storage.local.set(data);
  } else {
    return new Promise((resolve) => {
      chrome.storage.local.set(data, resolve);
    });
  }
}

// Initialize AI models
async function initializeModels() {
  try {
    console.log('Loading AI models...');
    
    // Configure Transformers.js for browser environment
    env.allowLocalModels = false;
    env.useBrowserCache = true;
    
    // Load Qwen3-Embedding model for feature extraction
    console.log('Loading Qwen3-Embedding model...');
    embeddingPipeline = await pipeline('feature-extraction', 'onnx-community/Qwen3-Embedding-0.6B-ONNX');
    
    // Load custom classifier model
    console.log('Loading custom classifier model...');
    const modelResponse = await fetch(MODEL_PATH);
    const modelArrayBuffer = await modelResponse.arrayBuffer();
    const modelUint8Array = new Uint8Array(modelArrayBuffer);
    
    classifierSession = await ort.InferenceSession.create(modelUint8Array);
    
    isModelsLoaded = true;
    console.log('AI models loaded successfully');
    
    // Update storage to indicate models are ready
    await setStorageData({ 
      modelsLoaded: true, 
      lastModelLoad: Date.now(),
      embeddingModelLoaded: !!embeddingPipeline,
      classifierModelLoaded: !!classifierSession
    });
    
  } catch (error: any) {
    console.error('Failed to load AI models:', error);
    await setStorageData({ 
      modelsLoaded: false, 
      modelError: error?.message || 'Unknown error' 
    });
  }
}

// Classify JavaScript content using AI models
async function classifyContent(content: string): Promise<number> {
  if (!isModelsLoaded || !embeddingPipeline || !classifierSession) {
    return 0.0;
  }
  
  try {
    // Extract embeddings using Qwen3
    const embeddingResult = await embeddingPipeline(content, { 
      pooling: 'mean', 
      normalize: true 
    });
    
    // Get the embedding tensor (should be 1024-dim vector)
    const embedding = embeddingResult.data;
    
    // Prepare input for ONNX classifier
    const inputTensor = new ort.Tensor('float32', embedding, [1, 1024]);
    
    // Run inference
    const results = await classifierSession.run({ 'input': inputTensor });
    const probability = results['output'].data[0] as number; // Assuming output is named 'output'
    
    return Math.min(Math.max(probability, 0.0), 1.0); // Clamp between 0 and 1
    
  } catch (error: any) {
    console.error('Classification error:', error);
    return 0.0;
  }
}

// Fetch and analyze JavaScript content
async function analyzeJavaScript(url: string): Promise<boolean> {
  try {
    const response = await fetch(url, { 
      method: 'GET',
      headers: { 'Accept': 'application/javascript, text/javascript' }
    });
    
    if (!response.ok) return false;
    
    const content = await response.text();
    
    // Skip if content is too large (limit to 50KB for performance)
    if (content.length > 50000) {
      return false;
    }
    
    totalRequests++;
    
    // Classify the content
    const probability = await classifyContent(content);
    
    if (probability > BLOCK_THRESHOLD) {
      blockedRequests++;
      console.log(`Blocked ${url} - Probability: ${probability.toFixed(3)}`);
      
      // Update statistics
      await setStorageData({
        blockedRequests,
        totalRequests,
        lastBlocked: Date.now()
      });
      
      return true;
    }
    
    return false;
    
  } catch (error: any) {
    console.error(`Failed to analyze ${url}:`, error);
    return false;
  }
}

// Request interception setup
async function setupRequestInterception() {
  if (isFirefox) {
    // Firefox: Use webRequest API
    browserAPI.webRequest.onBeforeRequest.addListener(
      async (details: any) => {
        if (details.type === 'script' && isModelsLoaded) {
          const shouldBlock = await analyzeJavaScript(details.url);
          if (shouldBlock) {
            return { cancel: true };
          }
        }
        return { cancel: false };
      },
      { urls: ['<all_urls>'] },
      ['blocking']
    );
  } else {
    // Chrome: Use declarativeNetRequest with dynamic rules
    try {
      // Remove existing rules
      const existingRules = await chrome.declarativeNetRequest.getDynamicRules();
      if (existingRules.length > 0) {
        await chrome.declarativeNetRequest.updateDynamicRules({
          removeRuleIds: existingRules.map((rule: any) => rule.id)
        });
      }
      
      // Set up listener for real-time classification
      chrome.webRequest.onBeforeRequest.addListener(
        async (details: any) => {
          if (details.type === 'script' && isModelsLoaded) {
            const shouldBlock = await analyzeJavaScript(details.url);
            if (shouldBlock) {
              console.log(`Blocking request: ${details.url}`);
              return { cancel: true };
            }
          }
          return { cancel: false };
        },
        { urls: ['<all_urls>'] },
        ['blocking']
      );
      
    } catch (error: any) {
      console.error('Chrome request interception setup failed:', error);
    }
  }
}

// Extension initialization
async function initializeExtension() {
  console.log('Initializing μDeep Blocker...');
  
  // Load existing statistics
  const storage = await getStorageData(['blockedRequests', 'totalRequests']);
  blockedRequests = storage.blockedRequests || 0;
  totalRequests = storage.totalRequests || 0;
  
  // Initialize AI models
  await initializeModels();
  
  // Set up request interception
  await setupRequestInterception();
  
  console.log('μDeep Blocker initialized successfully');
}

// Listen for extension startup
if (isFirefox) {
  browserAPI.runtime.onStartup.addListener(initializeExtension);
  browserAPI.runtime.onInstalled.addListener(initializeExtension);
} else {
  chrome.runtime.onStartup.addListener(initializeExtension);
  chrome.runtime.onInstalled.addListener(initializeExtension);
}

// Handle messages from popup
if (isFirefox) {
  browserAPI.runtime.onMessage.addListener((message: any, _sender: any, sendResponse: any) => {
    if (message.action === 'getStats') {
      sendResponse({
        blockedRequests,
        totalRequests,
        modelsLoaded: isModelsLoaded
      });
    } else if (message.action === 'getModelsStatus') {
      sendResponse({
        loaded: isModelsLoaded,
        embeddingModel: !!embeddingPipeline,
        classifierModel: !!classifierSession
      });
    } else if (message.action === 'updateThreshold') {
      // Update threshold logic would go here
      sendResponse({ success: true });
    }
  });
} else {
  chrome.runtime.onMessage.addListener((message: any, _sender: any, sendResponse: any) => {
    if (message.action === 'getStats') {
      sendResponse({
        blockedRequests,
        totalRequests,
        modelsLoaded: isModelsLoaded
      });
    } else if (message.action === 'getModelsStatus') {
      sendResponse({
        loaded: isModelsLoaded,
        embeddingModel: !!embeddingPipeline,
        classifierModel: !!classifierSession
      });
    } else if (message.action === 'updateThreshold') {
      // Update threshold logic would go here
      sendResponse({ success: true });
    }
    return true; // Keep message channel open for async response
  });
}

// Keep service worker alive (Chrome MV3 requirement)
if (!isFirefox) {
  setInterval(() => {
    // Ping to keep service worker alive
  }, 25000);
}
