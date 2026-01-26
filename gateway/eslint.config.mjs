// @ts-check
import eslint from '@eslint/js';
import eslintPluginPrettierRecommended from 'eslint-plugin-prettier/recommended';
import globals from 'globals';
import tseslint from 'typescript-eslint';

export default tseslint.config(
  {
    ignores: [
      // Config files (anywhere)
      '**/eslint.config.mjs',
      '**/jest.config.js',
      '**/jest.setup.ts',
      // Build outputs (anywhere)
      '**/dist/**',
      '**/build/**',
      '**/.next/**',
      '**/.turbo/**',
      '**/out/**',
      // Dependencies (anywhere)
      '**/node_modules/**',
      // Python virtual envs
      '**/.venv/**',
      '**/venv/**',
      '**/__pycache__/**',
      // Test coverage
      '**/coverage/**',
      // Generated/minified files
      '**/*.min.js',
      '**/*.min.mjs',
      '**/*.map',
      '**/*.d.ts',
      // Mocks
      '**/__mocks__/**',
      // Lock files
      '**/pnpm-lock.yaml',
      '**/package-lock.json',
    ],
  },
  eslint.configs.recommended,
  ...tseslint.configs.recommendedTypeChecked,
  eslintPluginPrettierRecommended,
  {
    languageOptions: {
      globals: {
        ...globals.node,
        ...globals.jest,
      },
      sourceType: 'commonjs',
      parserOptions: {
        projectService: true,
        tsconfigRootDir: import.meta.dirname,
      },
    },
  },
  {
    rules: {
      '@typescript-eslint/no-explicit-any': 'off',
      '@typescript-eslint/no-floating-promises': 'warn',
      '@typescript-eslint/no-unsafe-argument': 'warn',
      '@typescript-eslint/no-unsafe-assignment': 'warn',
      '@typescript-eslint/no-unsafe-member-access': 'warn',
      '@typescript-eslint/no-unsafe-return': 'warn',
      '@typescript-eslint/no-unsafe-call': 'warn',
      '@typescript-eslint/require-await': 'warn',
      '@typescript-eslint/no-unused-vars': [
        'warn',
        {
          argsIgnorePattern: '^_',
          varsIgnorePattern: '^_',
          caughtErrorsIgnorePattern: '^_|^error$|^err$|^e$',
        },
      ],
      '@typescript-eslint/no-require-imports': 'warn',
      '@typescript-eslint/prefer-promise-reject-errors': 'warn',
      '@typescript-eslint/no-misused-promises': 'warn',
    },
  },
);
