/*
 * Verify the Transformers.js local-model path used by the extension.
 * Run after packaging: node experiments/verify_browser_runtime.mjs
 */
import { env, pipeline } from '@xenova/transformers';

env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = './assets/models/';
env.useBrowserCache = false;

const classifier = await pipeline('text-classification', 'custom-macd-model', {
  top_k: null,
});
const result = await classifier('यह एक नया परीक्षण संदेश है');
if (!Array.isArray(result) || result.length === 0) {
  throw new Error('Transformers.js returned no classification result');
}
console.log(JSON.stringify({
  runtime: '@xenova/transformers',
  model: 'custom-macd-model',
  result,
  passed: true,
}));
