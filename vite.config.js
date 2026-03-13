import { defineConfig } from 'vite';
import { crx } from '@crxjs/vite-plugin';
import manifest from './manifest.json';
import path from 'path';
import fs from 'fs';

// Custom plugin to copy WASM + model assets into the dist folder
function copyAssetsPlugin() {
  return {
    name: 'copy-assets',
    closeBundle() {
      const distDir = path.resolve(__dirname, 'dist');

      // 1. Copy WASM files from @xenova/transformers/dist into dist/assets/
      const wasmSrc = path.resolve(__dirname, 'node_modules/@xenova/transformers/dist');
      const wasmDest = path.resolve(distDir, 'assets');
      if (!fs.existsSync(wasmDest)) fs.mkdirSync(wasmDest, { recursive: true });
      const wasmFiles = fs.readdirSync(wasmSrc).filter(f => f.endsWith('.wasm'));
      for (const file of wasmFiles) {
        fs.copyFileSync(path.join(wasmSrc, file), path.join(wasmDest, file));
        console.log(`[copy-assets] Copied WASM: ${file}`);
      }

      // 2. Copy the model folder into dist/assets/models/
      const modelSrc = path.resolve(__dirname, 'assets/models');
      const modelDest = path.resolve(distDir, 'assets/models');
      copyDirRecursive(modelSrc, modelDest);
      console.log('[copy-assets] Model assets copied to dist/assets/models/');
    }
  };
}

function copyDirRecursive(src, dest) {
  if (!fs.existsSync(dest)) fs.mkdirSync(dest, { recursive: true });
  for (const entry of fs.readdirSync(src, { withFileTypes: true })) {
    const srcPath = path.join(src, entry.name);
    const destPath = path.join(dest, entry.name);
    if (entry.isDirectory()) {
      copyDirRecursive(srcPath, destPath);
    } else {
      fs.copyFileSync(srcPath, destPath);
    }
  }
}

export default defineConfig({
  plugins: [crx({ manifest }), copyAssetsPlugin()],
  server: {
    port: 5173,
    strictPort: true,
    hmr: {
      port: 5173,
    },
  },
});
