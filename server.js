require('dotenv').config();
const express = require('express');
const multer = require('multer');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const pdfParse = require('pdf-parse');
const mammoth = require('mammoth');
const crypto = require('crypto');

const app = express();
app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.static('public'));

const upload = multer({ dest: 'uploads/' });

// 通义千问配置 (OpenAI 兼容模式)
const DASHSCOPE_API_KEY = process.env.DASHSCOPE_API_KEY;
const BASE_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1';
const TEXT_MODEL = 'qwen3.6-plus';          // 纯文本模型
const VISION_MODEL = 'qwen-vl-max';         // 视觉模型（识别图片文字）
// 也可用 qwen-vl-plus，更便宜，根据你的需求修改

// 任务存储
const tasks = new Map();

// ==================== 核心工具函数 ====================

// 纯文本调用
async function callQWen(messages, temperature = 0.3) {
  const url = `${BASE_URL}/chat/completions`;
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 300000); // 5分钟超时

  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${DASHSCOPE_API_KEY}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ model: TEXT_MODEL, messages, temperature }),
      signal: controller.signal
    });
    clearTimeout(timeoutId);

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.error?.message || `状态码 ${response.status}`);
    }
    const data = await response.json();
    if (data.choices?.[0]?.message) {
      return data.choices[0].message.content.trim();
    }
    throw new Error('API 返回格式异常');
  } catch (e) {
    clearTimeout(timeoutId);
    if (e.name === 'AbortError') throw new Error('模型响应超时，请稍后重试');
    throw e;
  }
}

// 视觉模型调用（支持图片）
async function callVision(messages, maxTokens = 4096) {
  const url = `${BASE_URL}/chat/completions`;
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 300000);

  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${DASHSCOPE_API_KEY}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model: VISION_MODEL,
        messages,
        max_tokens: maxTokens
      }),
      signal: controller.signal
    });
    clearTimeout(timeoutId);

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.error?.message || `状态码 ${response.status}`);
    }
    const data = await response.json();
    if (data.choices?.[0]?.message?.content) {
      return data.choices[0].message.content.trim();
    }
    throw new Error('视觉模型返回格式异常');
  } catch (e) {
    clearTimeout(timeoutId);
    if (e.name === 'AbortError') throw new Error('视觉模型响应超时，请稍后重试');
    throw e;
  }
}

// 图片转 base64
function imageToBase64(filePath) {
  const buffer = fs.readFileSync(filePath);
  const ext = path.extname(filePath).slice(1).toLowerCase();
  const mimeMap = {
    jpg: 'image/jpeg',
    jpeg: 'image/jpeg',
    png: 'image/png',
    webp: 'image/webp',
    bmp: 'image/bmp'
  };
  const mime = mimeMap[ext] || 'image/png';
  return `data:${mime};base64,${buffer.toString('base64')}`;
}

// ✅ 用视觉模型提取图片文字（不再使用 Tesseract，稳定可靠）
async function extractTextFromImage(filePath) {
  const imageBase64 = imageToBase64(filePath);

  const messages = [
    {
      role: 'user',
      content: [
        {
          type: 'image_url',
          image_url: { url: imageBase64 }
        },
        {
          type: 'text',
          text: '请提取这张图片中的所有文字内容，保持原有格式和顺序，直接输出文字，不要添加任何解释。'
        }
      ]
    }
  ];

  const text = await callVision(messages, 4096);
  if (!text) throw new Error('视觉模型未返回任何文字');
  return text;
}

// PDF / Word / TXT 提取（不变）
async function extractTextFromPDF(filePath) {
  const data = await pdfParse(fs.readFileSync(filePath));
  return data.text;
}

async function extractTextFromWord(filePath) {
  const result = await mammoth.extractRawText({ path: filePath });
  return result.value;
}

// 本地关键词情感分析 (降级方案)
function localSentiment(text) {
  const positive = ['哈哈', '开心', '喜欢', '爱', '好', '棒', '快乐', '幸福', '😊', '😄', '👍'];
  const negative = ['难过', '生气', '讨厌', '恨', '不好', '烦', '哭', '😢', '😡', '👎'];
  let score = 0;
  positive.forEach(w => { if (text.includes(w)) score += 0.3; });
  negative.forEach(w => { if (text.includes(w)) score -= 0.3; });
  const valence = Math.max(-1, Math.min(1, score));
  const arousal = Math.abs(valence) * 0.8 + 0.2;
  const emotion = valence > 0.2 ? 'joy' : (valence < -0.2 ? 'sadness' : 'neutral');
  return { valence, arousal, emotion };
}

// ==================== API 接口 ====================

// 1. 发起解析任务
app.post('/api/parse', upload.array('files', 10), async (req, res) => {
  try {
    let rawText = '';
    for (let file of req.files) {
      let text = '';
      if (file.mimetype.startsWith('image/')) {
        text = await extractTextFromImage(file.path);   // 视觉模型 OCR
      } else if (file.mimetype === 'application/pdf') {
        text = await extractTextFromPDF(file.path);
      } else if (file.mimetype.includes('word') || file.mimetype.includes('document')) {
        text = await extractTextFromWord(file.path);
      } else if (file.mimetype === 'text/plain') {
        text = fs.readFileSync(file.path, 'utf-8');
      }
      rawText += text + '\n';
    }

    if (!rawText.trim()) {
      return res.json({ success: false, error: '未能提取到文本内容，请检查文件' });
    }

    const taskId = crypto.randomUUID();
    const task = {
      id: taskId,
      status: 'parsing',
      progress: 0,
      message: '开始解析对话...',
      startTime: Date.now(),
      parsedMessages: [],
      result: null,
      error: null
    };
    tasks.set(taskId, task);

    // 异步启动任务，不阻塞响应
    processTask(taskId, rawText).catch(e => {
      task.status = 'error';
      task.error = e.message;
      console.error('任务失败:', e);
    });

    res.json({ success: true, taskId });
  } catch (e) {
    res.json({ success: false, error: e.message });
  }
});

// 2. 查询任务进度
app.get('/api/task/:taskId/progress', (req, res) => {
  const task = tasks.get(req.params.taskId);
  if (!task) return res.status(404).json({ error: '任务不存在' });

  const elapsed = (Date.now() - task.startTime) / 1000;
  let estimate = null;
  if (task.progress > 0 && task.status !== 'completed' && task.status !== 'error') {
    const remaining = 100 - task.progress;
    const speed = task.progress / elapsed;
    if (speed > 0) estimate = Math.round(remaining / speed);
  }

  res.json({
    taskId: task.id,
    status: task.status,
    progress: task.progress,
    message: task.message,
    estimate,
    parsedCount: task.parsedMessages.length
  });
});

// 3. 获取最终结果
app.get('/api/task/:taskId/result', (req, res) => {
  const task = tasks.get(req.params.taskId);
  if (!task) return res.status(404).json({ error: '任务不存在' });
  if (task.status !== 'completed') {
    return res.status(400).json({ error: '任务尚未完成', status: task.status });
  }
  res.json({ success: true, data: task.result, speakers: [...new Set(task.result.map(m => m.speaker))] });
});

// 4. 模拟聊天接口
app.post('/api/chat', async (req, res) => {
  const { message, history, partnerProfile } = req.body;
  if (!history || history.length < 10) return res.json({ reply: '（数据不足）' });
  const context = history.map(m => `${m.speaker}: ${m.text}`).join('\n');
  const prompt = `模仿此人回复：${JSON.stringify(partnerProfile)}。\n对话：\n${context}\n用户：“${message}”\n请以对方语气回复，仅返回回复内容。`;
  try {
    const reply = await callQWen([{ role: 'user', content: prompt }], 0.8);
    res.json({ reply });
  } catch (e) {
    res.json({ reply: '（暂时无法回应）' });
  }
});

// ==================== 核心任务处理 (批量情感分析) ====================
async function processTask(taskId, rawText) {
  const task = tasks.get(taskId);
  if (!task) return;

  try {
    // 阶段1: 分片解析对话结构
    const CHUNK_SIZE = 3000;
    const chunks = [];
    for (let i = 0; i < rawText.length; i += CHUNK_SIZE) {
      chunks.push(rawText.slice(i, i + CHUNK_SIZE));
    }
    const totalChunks = chunks.length;
    task.message = `正在解析对话结构 (共 ${totalChunks} 段)...`;
    task.progress = 0;

    let allParsed = [];
    for (let i = 0; i < chunks.length; i++) {
      const prompt = `解析聊天记录（第${i+1}/${totalChunks}段），返回JSON数组，元素格式：{"speaker":"我或对方","text":"...","time":"..."}。只返回JSON数组。文本：\n${chunks[i]}`;
      const text = await callQWen([{ role: 'user', content: prompt }], 0.3);
      let cleaned = text.replace(/^```(?:json)?\s*\n?/i, '').replace(/\n?```\s*$/i, '');
      try {
        const arr = JSON.parse(cleaned);
        if (Array.isArray(arr)) allParsed = allParsed.concat(arr);
      } catch(e) {}
      task.progress = Math.round((i + 1) / totalChunks * 25);
      task.parsedMessages = allParsed;
      task.message = `解析对话 ${i+1}/${totalChunks} 段...`;
    }

    if (allParsed.length === 0) throw new Error('未解析到任何有效消息');

    // 去重
    const unique = [];
    for (let i = 0; i < allParsed.length; i++) {
      if (i === 0 || allParsed[i].text !== allParsed[i-1].text || allParsed[i].speaker !== allParsed[i-1].speaker) {
        unique.push(allParsed[i]);
      }
    }

    // 阶段2: 批量情感分析
    task.status = 'analyzing';
    task.message = '正在进行批量情感分析...';
    const total = unique.length;
    task.progress = 25;
    let enriched = [];

    const BATCH_SIZE = 10; // 每批10条，避免超时
    let processed = 0;
    for (let i = 0; i < unique.length; i += BATCH_SIZE) {
      const batch = unique.slice(i, i + BATCH_SIZE);
      const batchMessages = JSON.stringify(batch.map(m => m.text));
      const batchPrompt = `对以下每条消息进行情感分析，返回一个JSON数组，长度必须等于${batch.length}，每个元素格式：{"valence":-1~1,"arousal":0~1,"emotion":"joy/sadness/anger/surprise/love/anxiety"}。只返回数组。\n消息列表：${batchMessages}`;

      let batchResult = [];
      for (let attempt = 0; attempt < 2; attempt++) {
        try {
          const resultText = await callQWen([{ role: 'user', content: batchPrompt }], 0);
          let cleaned = resultText.replace(/^```(?:json)?\s*\n?/i, '').replace(/\n?```\s*$/i, '');
          const arr = JSON.parse(cleaned);
          if (Array.isArray(arr)) {
            batchResult = arr;
            break;
          }
        } catch(e) {
          console.error(`情感分析批 ${Math.floor(i/BATCH_SIZE)+1} 尝试${attempt+1} 失败:`, e.message);
          if (attempt === 1) {
            batchResult = batch.map(m => localSentiment(m.text));
          }
        }
      }

      // 补齐长度
      while (batchResult.length < batch.length) {
        batchResult.push({ valence: 0, arousal: 0.5, emotion: 'neutral' });
      }
      for (let j = 0; j < batch.length; j++) {
        const msg = batch[j];
        const sent = batchResult[j] || { valence: 0, arousal: 0.5, emotion: 'neutral' };
        enriched.push({
          ...msg,
          ...sent,
          date: msg.time || new Date().toISOString().slice(0,10)
        });
      }
      processed += batch.length;
      task.progress = 25 + Math.round((processed / total) * 75);
      task.message = `情感分析 ${processed}/${total} 条消息 (批量处理中)...`;
    }

    task.status = 'completed';
    task.progress = 100;
    task.result = enriched;
    task.message = '解析完成';
  } catch (e) {
    task.status = 'error';
    task.error = e.message;
    task.message = '解析失败: ' + e.message;
  }
}

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => console.log(`✅ 服务已启动 → http://localhost:${PORT}`));