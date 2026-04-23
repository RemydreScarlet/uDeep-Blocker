import { defineConfig } from 'vite';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = fileURLToPath(new URL('.', import.meta.url));

export default defineConfig(({ mode }) => {
  const isFirefox = mode === 'firefox';
  const isChrome = mode === 'chrome' || !isFirefox;

  return {
    build: {
      outDir: 'dist',
      emptyOutDir: true,
      sourcemap: true,
      minify: false,
      target: 'es2022',
      rollupOptions: {
        input: {
          'background': resolve(__dirname, 'src/background.ts'),
          'content': resolve(__dirname, 'src/content.ts'),
          'popup/public/popup': resolve(__dirname, 'public/popup.html')
        },
        output: {
          entryFileNames: '[name].js',
          chunkFileNames: '[name].js',
          assetFileNames: '[name].[ext]',
          format: 'es'
        }
      }
    },

    resolve: {
      alias: {
        '@': resolve(__dirname, 'src'),
        '~': resolve(__dirname, 'public')
      }
    },

    define: {
      __BROWSER__: JSON.stringify(isFirefox ? 'firefox' : 'chrome'),
      __IS_FIREFOX__: isFirefox,
      __IS_CHROME__: isChrome
    }
  };
});
