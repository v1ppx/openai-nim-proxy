// server.js - OpenAI to NVIDIA NIM API Proxy for Vercel

const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();

// Middleware
app.use(cors());
app.use(express.json({ limit: '10mb' }));

// NVIDIA NIM configuration
const NIM_API_BASE =
  process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1/chat/completions';

const NIM_API_KEY = process.env.NIM_API_KEY;

// Reasoning settings
const SHOW_REASONING = false;
const ENABLE_THINKING_MODE = false;

// Model mapping
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4': 'qwen/qwen3-coder-480b-a35b-instruct',
  'gpt-4-turbo': 'moonshotai/kimi-k2-instruct-0905',
  'gpt-4o': 'deepseek-ai/deepseek-v3.1',
  'claude-3-opus': 'openai/gpt-oss-120b',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro': 'qwen/qwen3-next-80b-a3b-thinking',

  // Gemma 4 31B
  'gemma-4-31b': 'google/gemma-4-31b-it',
  'google/gemma-4-31b-it': 'google/gemma-4-31b-it'
};

// Health check
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE,
    nim_configured: !!NIM_API_KEY
  });
});

// OpenAI-compatible model list
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map((model) => ({
    id: model,
    object: 'model',
    created: Math.floor(Date.now() / 1000),
    owned_by: 'nvidia-nim-proxy'
  }));

  res.json({
    object: 'list',
    data: models
  });
});

// Chat completions
app.post('/v1/chat/completions', async (req, res) => {
  try {
    if (!NIM_API_KEY) {
      return res.status(500).json({
        error: {
          message: 'NIM_API_KEY environment variable is not configured.',
          type: 'configuration_error',
          code: 500
        }
      });
    }

    const {
      model,
      messages,
      temperature,
      max_tokens,
      stream
    } = req.body;

    if (!messages || !Array.isArray(messages)) {
      return res.status(400).json({
        error: {
          message: 'messages must be an array.',
          type: 'invalid_request_error',
          code: 400
        }
      });
    }

    // Select model
    let nimModel = MODEL_MAPPING[model];

    // If Janitor sends an unrecognized model name,
    // use Gemma 4 31B as the default.
    if (!nimModel) {
      nimModel = 'google/gemma-4-31b-it';
    }

    // NVIDIA request
    const nimRequest = {
      model: nimModel,
      messages,
      temperature:
        typeof temperature === 'number' ? temperature : 0.7,
      max_tokens:
        typeof max_tokens === 'number' ? max_tokens : 4096,
      stream: !!stream
    };

    // Thinking mode
    if (ENABLE_THINKING_MODE) {
      nimRequest.chat_template_kwargs = {
        thinking: true
      };
    }

    const response = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      nimRequest,
      {
        headers: {
          Authorization: `Bearer ${NIM_API_KEY}`,
          'Content-Type': 'application/json'
        },
        responseType: stream ? 'stream' : 'json',
        timeout: 120000,
        validateStatus: () => true
      }
    );

    // NVIDIA returned an error
    if (response.status < 200 || response.status >= 300) {
      let errorData = response.data;

      if (typeof errorData === 'string') {
        try {
          errorData = JSON.parse(errorData);
        } catch {
          // Leave it as a string
        }
      }

      return res.status(response.status).json(
        errorData || {
          error: {
            message: `NVIDIA NIM returned HTTP ${response.status}`,
            type: 'api_error',
            code: response.status
          }
        }
      );
    }

    // Streaming response
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      let buffer = '';
      let reasoningStarted = false;

      response.data.on('data', (chunk) => {
        buffer += chunk.toString();

        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) {
            continue;
          }

          if (line.includes('[DONE]')) {
            res.write('data: [DONE]\n\n');
            continue;
          }

          try {
            const data = JSON.parse(line.slice(6));
            const delta = data.choices?.[0]?.delta;

            if (delta) {
              const reasoning = delta.reasoning_content;
              const content = delta.content;

              if (SHOW_REASONING) {
                let combined = '';

                if (reasoning && !reasoningStarted) {
                  combined += '<think>\n';
                  reasoningStarted = true;
                }

                if (reasoning) {
                  combined += reasoning;
                }

                if (content && reasoningStarted) {
                  combined += '</think>\n\n';
                  combined += content;
                  reasoningStarted = false;
                } else if (content) {
                  combined += content;
                }

                if (combined) {
                  delta.content = combined;
                }

                delete delta.reasoning_content;
              } else {
                delta.content = content || '';
                delete delta.reasoning_content;
              }
            }

            res.write(`data: ${JSON.stringify(data)}\n\n`);
          } catch {
            // Ignore malformed streaming chunks
          }
        }
      });

      response.data.on('end', () => {
        res.end();
      });

      response.data.on('error', (error) => {
        console.error('NVIDIA stream error:', error);
        res.end();
      });

      return;
    }

    // Normal non-streaming response
    const data = response.data;

    const openaiResponse = {
      id: data.id || `chatcmpl-${Date.now()}`,
      object: 'chat.completion',
      created: data.created || Math.floor(Date.now() / 1000),
      model: model || nimModel,
      choices: (data.choices || []).map((choice) => {
        let content = choice.message?.content || '';

        if (
          SHOW_REASONING &&
          choice.message?.reasoning_content
        ) {
          content =
            '<think>\n' +
            choice.message.reasoning_content +
            '\n</think>\n\n' +
            content;
        }

        return {
          index: choice.index ?? 0,
          message: {
            role: choice.message?.role || 'assistant',
            content
          },
          finish_reason: choice.finish_reason || 'stop'
        };
      }),
      usage: data.usage || {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0
      }
    };

    return res.json(openaiResponse);

  } catch (error) {
    console.error(
      'Proxy error:',
      error.response?.data || error.message
    );

    return res.status(error.response?.status || 500).json({
      error: {
        message:
          error.response?.data?.error?.message ||
          error.message ||
          'Internal server error',
        type: 'invalid_request_error',
        code: error.response?.status || 500
      }
    });
  }
});

// Root endpoint
app.get('/', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    message: 'Proxy is running.'
  });
});

// 404 handler
app.use((req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

// IMPORTANT:
// Do NOT use app.listen() on Vercel.
// Vercel handles the HTTP server.
module.exports = app;
