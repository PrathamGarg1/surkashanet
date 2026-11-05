import { defineConfig } from 'vite';
import { crx } from '@crxjs/vite-plugin';
import manifest from './manifest.json' with { type: 'json' };
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

function copyAssetsPlugin() {
  return {
    name: 'copy-assets',
    closeBundle() {
      const distDir = path.resolve(__dirname, 'dist');
      const wasmSrc = path.resolve(__dirname, 'node_modules/@xenova/transformers/dist');
      const wasmDest = path.resolve(distDir, 'assets');
      fs.mkdirSync(wasmDest, { recursive: true });
      for (const file of fs.readdirSync(wasmSrc).filter((f) => f.endsWith('.wasm'))) {
        fs.copyFileSync(path.join(wasmSrc, file), path.join(wasmDest, file));
      }
      const modelSrc = path.resolve(__dirname, 'assets/models');
      const modelDest = path.resolve(distDir, 'assets/models');
      fs.cpSync(modelSrc, modelDest, { recursive: true });
    },
  };
}

export default defineConfig({
  plugins: [crx({ manifest }), copyAssetsPlugin()],
});
