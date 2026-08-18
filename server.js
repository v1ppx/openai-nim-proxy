// server.js - OpenAI -> NVIDIA NIM Proxy for Vercel

const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();

app.use(cors());
app.use(express.json({ limit: '10mb' }));

const NIM_API_BASE =
  process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';

const NIM_API_KEY = process.env.NIM_API_KEY;

const SHOW_REASONING = false;
const ENABLE_THINKING_MODE = false;

const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4': 'qwen/qwen3-coder-480b-a35b-instruct',
  'gpt-4-turbo': 'moonshotai/kimi-k2-instruct-0905',
  'gpt-4o': 'deepseek-ai/deepseek-v3.1',
  'claude-3-opus': 'openai/gpt-oss-120b',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro': 'qwen/qwen3-next-80b-a3b-thinking',

  'gemma-4-31b': 'google/gemma-4-31b-it',
  'google/gemma-4-31b-it': 'google/gemma-4-31b-it'
};


// ---------------------------------------------------------
// HEALTH
// ---------------------------------------------------------

app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE,
    nim_configured: !!NIM_API_KEY
  });
});


// ---------------------------------------------------------
// MODELS
// ---------------------------------------------------------

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


// ---------------------------------------------------------
// CHAT COMPLETIONS
// ---------------------------------------------------------

app.post('/v1/chat/completions', async (req, res) => {

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

  // -------------------------------------------------------
  // MODEL
  // -------------------------------------------------------

  const nimModel =
    MODEL_MAPPING[model] ||
    'google/gemma-4-31b-it';


  // -------------------------------------------------------
  // NVIDIA REQUEST
  // -------------------------------------------------------

  const nimRequest = {
    model: nimModel,
    messages,
    temperature:
      typeof temperature === 'number'
        ? temperature
        : 0.7,

    max_tokens:
      typeof max_tokens === 'number'
        ? max_tokens
        : 4096,

    stream: !!stream
  };


  if (ENABLE_THINKING_MODE) {
    nimRequest.chat_template_kwargs = {
      thinking: true
    };
  }


  console.log(
    `NVIDIA request: model=${nimModel}, stream=${!!stream}, max_tokens=${nimRequest.max_tokens}`
  );


  try {

    // -----------------------------------------------------
    // STREAMING
    // -----------------------------------------------------

    if (stream) {

      const response = await axios.post(
        `${NIM_API_BASE}/chat/completions`,
        nimRequest,
        {
          headers: {
            Authorization: `Bearer ${NIM_API_KEY}`,
            'Content-Type': 'application/json',
            Accept: 'text/event-stream'
          },

          responseType: 'stream',

          // IMPORTANT:
          // Do not kill the NVIDIA request after 120 seconds.
          timeout: 0,

          validateStatus: () => true
        }
      );


      // NVIDIA error
      if (
        response.status < 200 ||
        response.status >= 300
      ) {

        let errorBody = '';

        response.data.on('data', chunk => {
          errorBody += chunk.toString();
        });

        response.data.on('end', () => {

          let parsed;

          try {
            parsed = JSON.parse(errorBody);
          } catch {
            parsed = {
              error: {
                message: errorBody ||
                  `NVIDIA NIM returned HTTP ${response.status}`,
                type: 'api_error',
                code: response.status
              }
            };
          }

          if (!res.headersSent) {
            res.status(response.status).json(parsed);
          } else {
            res.end();
          }

        });

        return;
      }


      // ---------------------------------------------------
      // SSE HEADERS
      // ---------------------------------------------------

      res.status(200);

      res.setHeader(
        'Content-Type',
        'text/event-stream; charset=utf-8'
      );

      res.setHeader(
        'Cache-Control',
        'no-cache, no-transform'
      );

      res.setHeader(
        'Connection',
        'keep-alive'
      );

      res.setHeader(
        'X-Accel-Buffering',
        'no'
      );


      // Send headers immediately
      if (res.flushHeaders) {
        res.flushHeaders();
      }


      // ---------------------------------------------------
      // FORWARD NVIDIA STREAM
      // ---------------------------------------------------

      let buffer = '';
      let reasoningStarted = false;

      response.data.on('data', chunk => {

        buffer += chunk.toString();

        const lines = buffer.split('\n');

        buffer = lines.pop() || '';

        for (const line of lines) {

          if (!line.startsWith('data:')) {
            continue;
          }

          const payload =
            line.slice(5).trim();

          if (!payload) {
            continue;
          }


          // NVIDIA finished
          if (payload === '[DONE]') {

            res.write('data: [DONE]\n\n');

            continue;
          }


          try {

            const data =
              JSON.parse(payload);

            const delta =
              data.choices?.[0]?.delta;


            if (delta) {

              const reasoning =
                delta.reasoning_content;

              const content =
                delta.content;


              if (SHOW_REASONING) {

                let combined = '';

                if (
                  reasoning &&
                  !reasoningStarted
                ) {
                  combined += '<think>\n';
                  reasoningStarted = true;
                }

                if (reasoning) {
                  combined += reasoning;
                }

                if (
                  content &&
                  reasoningStarted
                ) {
                  combined +=
                    '</think>\n\n';

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

                delta.content =
                  content || '';

                delete delta.reasoning_content;

              }

            }


            // Immediately send chunk to Janitor
            res.write(
              `data: ${JSON.stringify(data)}\n\n`
            );


          } catch (err) {

            console.error(
              'SSE parse error:',
              err.message
            );

          }

        }

      });


      // NVIDIA stream finished
      response.data.on('end', () => {

        if (!res.writableEnded) {
          res.end();
        }

      });


      // NVIDIA stream error
      response.data.on('error', error => {

        console.error(
          'NVIDIA stream error:',
          error.message
        );

        if (!res.writableEnded) {
          res.end();
        }

      });


      // Client disconnected
      req.on('close', () => {

        if (!res.writableEnded) {

          console.log(
            'Client disconnected.'
          );

          response.data.destroy();

        }

      });


      return;
    }


    // -----------------------------------------------------
    // NON-STREAMING REQUEST
    // -----------------------------------------------------

    const response = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      nimRequest,
      {
        headers: {
          Authorization: `Bearer ${NIM_API_KEY}`,
          'Content-Type': 'application/json'
        },

        responseType: 'json',

        timeout: 0,

        validateStatus: () => true
      }
    );


    if (
      response.status < 200 ||
      response.status >= 300
    ) {

      return res.status(response.status).json(
        response.data || {
          error: {
            message:
              `NVIDIA NIM returned HTTP ${response.status}`,
            type: 'api_error',
            code: response.status
          }
        }
      );

    }


    const data = response.data;


    const openaiResponse = {
      id:
        data.id ||
        `chatcmpl-${Date.now()}`,

      object:
        'chat.completion',

      created:
        data.created ||
        Math.floor(Date.now() / 1000),

      model:
        model ||
        nimModel,

      choices:
        (data.choices || []).map(choice => {

          let content =
            choice.message?.content || '';


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

            index:
              choice.index ?? 0,

            message: {
              role:
                choice.message?.role ||
                'assistant',

              content
            },

            finish_reason:
              choice.finish_reason ||
              'stop'
          };

        }),

      usage:
        data.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
    };


    return res.json(openaiResponse);


  } catch (error) {

    console.error(
      'Proxy error:',
      error.code,
      error.message
    );


    if (error.code === 'ECONNABORTED') {

      return res.status(504).json({
        error: {
          message:
            'NVIDIA NIM request timed out.',
          type: 'timeout_error',
          code: 504
        }
      });

    }


    return res.status(
      error.response?.status || 500
    ).json({

      error: {

        message:
          error.response?.data?.error?.message ||
          error.message ||
          'Internal server error',

        type:
          'invalid_request_error',

        code:
          error.response?.status ||
          500

      }

    });

  }

});


// ---------------------------------------------------------
// ROOT
// ---------------------------------------------------------

// Root endpoint
app.get('/', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    message: 'Proxy is running.'
  });
});

// OpenAI-compatible /v1 endpoint
app.get('/v1', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    message: 'OpenAI-compatible API is running.',
    endpoints: {
      models: '/v1/models',
      chat_completions: '/v1/chat/completions'
    }
  });
});

// 404 handler
app.use((req, res) => {


// ---------------------------------------------------------
// 404
// ---------------------------------------------------------

app.use((req, res) => {

  res.status(404).json({
    error: {
      message:
        `Endpoint ${req.path} not found`,

      type:
        'invalid_request_error',

      code: 404
    }
  });

});


module.exports = app;
