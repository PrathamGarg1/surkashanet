import { pipeline, env } from '@xenova/transformers';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Configure environment for local models
env.allowLocalModels = true;
env.allowRemoteModels = false;
env.localModelPath = path.join(__dirname, 'assets', 'models');

console.log('Loading local model from:', env.localModelPath);

async function test() {
    try {
        const classifier = await pipeline('text-classification', 'custom-macd-model', {
            top_k: null
        });
        
        const texts = [
            "tu bahut achi hai", // Safe
            "teri maa ki aankh", // Abusive
            "kutta kahina", // Abusive
            "how are you today?" // Safe
        ];

        for (const text of texts) {
            console.log(`\nTesting: "${text}"`);
            const results = await classifier(text);
            console.log(results);
        }
    } catch (err) {
        console.error('Error during inference:', err);
    }
}

test();
