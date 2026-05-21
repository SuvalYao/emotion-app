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

// 使用内存存储，不再写入磁盘
const storage = multer.memoryStorage();
const upload = multer({ storage: storage });

// API 配置
const DASHSCOPE_API_KEY = process.env.DASHSCOPE_API_KEY;
const BASE_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1';
const TEXT_MODEL = 'qwen3.6-plus';
const VISION_MODEL = 'qwen-vl-max';

const tasks = new Map();

// ==================== 工具函数 ====================
async function callQWen(messages, temperature = 0.3) {
  const url = `${BASE_URL}/chat/completions`;
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 300000);
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${DASHSCOPE_API_KEY}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: TEXT_MODEL, messages, temperature }),
      signal: controller.signal
    });
    clearTimeout(timeoutId);
    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.error?.message || `状态码 ${response.status}`);
    }
    const data = await response.json();
    return data.choices?.[0]?.message?.content?.trim() || '';
  } catch (e) {
    clearTimeout(timeoutId);
    throw e;
  }
}

async function callVision(messages, maxTokens = 4096) {
  const url = `${BASE_URL}/chat/completions`;
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 300000);
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${DASHSCOPE_API_KEY}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: VISION_MODEL, messages, max_tokens: maxTokens }),
      signal: controller.signal
    });
    clearTimeout(timeoutId);
    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.error?.message || `状态码 ${response.status}`);
    }
    const data = await response.json();
    return data.choices?.[0]?.message?.content?.trim() || '';
  } catch (e) {
    clearTimeout(timeoutId);
    throw e;
  }
}

// 从 buffer 转 base64
function bufferToBase64(buffer, mimetype) {
  return `data:${mimetype};base64,${buffer.toString('base64')}`;
}

// 图片文字提取（视觉模型）
async function extractTextFromImage(buffer, mimetype) {
  const base64 = bufferToBase64(buffer, mimetype);
  const messages = [{
    role: 'user',
    content: [
      { type: 'image_url', image_url: { url: base64 } },
      { type: 'text', text: '请提取这张图片中的所有文字内容，保持原有格式和顺序，直接输出文字。' }
    ]
  }];
  return await callVision(messages, 4096);
}

// PDF 处理：buffer 写入临时文件再解析
async function extractTextFromPDF(buffer) {
  const tmpPath = path.join('/tmp', `${crypto.randomUUID()}.pdf`);
  fs.writeFileSync(tmpPath, buffer);
  const dataBuffer = fs.readFileSync(tmpPath);
  const data = await pdfParse(dataBuffer);
  fs.unlinkSync(tmpPath);
  return data.text;
}

// Word 处理：buffer 写入临时文件
async function extractTextFromWord(buffer) {
  const tmpPath = path.join('/tmp', `${crypto.randomUUID()}.docx`);
  fs.writeFileSync(tmpPath, buffer);
  const result = await mammoth.extractRawText({ path: tmpPath });
  fs.unlinkSync(tmpPath);
  return result.value;
}

// 本地情感降级
function localSentiment(text) {
  const positive = ['哈哈','开心','喜欢','爱','好','棒','快乐','幸福','😊','😄','👍'];
  const negative = ['难过','生气','讨厌','恨','不好','烦','哭','😢','😡','👎'];
  let score = 0;
  positive.forEach(w => { if(text.includes(w)) score+=0.3; });
  negative.forEach(w => { if(text.includes(w)) score-=0.3; });
  const valence = Math.max(-1, Math.min(1, score));
  const arousal = Math.abs(valence) * 0.8 + 0.2;
  const emotion = valence > 0.2 ? 'joy' : (valence < -0.2 ? 'sadness' : 'neutral');
  return { valence, arousal, emotion };
}

// ==================== 路由 ====================
app.post('/api/parse', upload.array('files', 10), async (req, res) => {
  try {
    let rawText = '';
    for (let file of req.files) {
      let text = '';
      if (file.mimetype.startsWith('image/')) {
        text = await extractTextFromImage(file.buffer, file.mimetype);
      } else if (file.mimetype === 'application/pdf') {
        text = await extractTextFromPDF(file.buffer);
      } else if (file.mimetype.includes('word') || file.mimetype.includes('document')) {
        text = await extractTextFromWord(file.buffer);
      } else if (file.mimetype === 'text/plain') {
        text = file.buffer.toString('utf-8');
      }
      rawText += text + '\n';
    }
    if (!rawText.trim()) return res.json({ success: false, error: '未提取到文本' });

    const taskId = crypto.randomUUID();
    const task = {
      id: taskId, status: 'parsing', progress: 0,
      message: '开始解析...', startTime: Date.now(),
      parsedMessages: [], result: null, error: null
    };
    tasks.set(taskId, task);
    processTask(taskId, rawText).catch(e => {
      task.status = 'error';
      task.error = e.message;
    });
    res.json({ success: true, taskId });
  } catch (e) {
    res.json({ success: false, error: e.message });
  }
});

app.get('/api/task/:taskId/progress', (req, res) => {
  const task = tasks.get(req.params.taskId);
  if (!task) return res.status(404).json({ error: '任务不存在' });
  const elapsed = (Date.now() - task.startTime)/1000;
  let estimate = null;
  if(task.progress>0 && task.status!=='completed' && task.status!=='error') {
    const remain = 100-task.progress;
    const speed = task.progress/elapsed;
    if(speed>0) estimate = Math.round(remain/speed);
  }
  res.json({
    taskId: task.id, status: task.status, progress: task.progress,
    message: task.message, estimate, parsedCount: task.parsedMessages.length
  });
});

app.get('/api/task/:taskId/result', (req, res) => {
  const task = tasks.get(req.params.taskId);
  if (!task) return res.status(404).json({ error: '任务不存在' });
  if (task.status !== 'completed') return res.status(400).json({ error: '任务未完成' });
  res.json({ success: true, data: task.result, speakers: [...new Set(task.result.map(m=>m.speaker))] });
});

app.post('/api/chat', async (req, res) => {
  const { message, history, partnerProfile } = req.body;
  if (!history || history.length < 10) return res.json({ reply: '数据不足' });
  const context = history.map(m => `${m.speaker}: ${m.text}`).join('\n');
  const prompt = `模仿此人：${JSON.stringify(partnerProfile)}。\n对话：\n${context}\n用户：“${message}”\n请以对方语气回复，仅返回回复内容。`;
  try {
    const reply = await callQWen([{ role: 'user', content: prompt }], 0.8);
    res.json({ reply });
  } catch(e) { res.json({ reply: '（暂时无法回应）' }); }
});

// ==================== 任务处理（批量情感分析） ====================
async function processTask(taskId, rawText) {
  const task = tasks.get(taskId);
  if (!task) return;
  try {
    const CHUNK_SIZE = 3000;
    const chunks = [];
    for (let i = 0; i < rawText.length; i += CHUNK_SIZE)
      chunks.push(rawText.slice(i, i+CHUNK_SIZE));
    task.message = `解析对话结构 (${chunks.length}段)`;
    let allParsed = [];
    for (let i=0; i<chunks.length; i++) {
      const prompt = `解析聊天记录（第${i+1}段），返回JSON数组，格式：[{"speaker":"我或对方","text":"...","time":"..."}]。文本：\n${chunks[i]}`;
      const txt = await callQWen([{ role:'user', content: prompt }], 0.3);
      let cleaned = txt.replace(/^```(?:json)?\s*\n?/i,'').replace(/\n?```\s*$/i,'');
      try { const arr=JSON.parse(cleaned); if(Array.isArray(arr)) allParsed=allParsed.concat(arr); } catch(e){}
      task.progress = Math.round((i+1)/chunks.length*25);
      task.parsedMessages = allParsed;
    }
    const unique = [];
    for (let i=0; i<allParsed.length; i++)
      if(i===0 || allParsed[i].text!==allParsed[i-1].text) unique.push(allParsed[i]);

    // 批量情感分析
    task.status='analyzing'; task.message='情感分析中...';
    const total = unique.length; let enriched=[];
    const BATCH_SIZE = 10; let processed=0;
    for (let i=0; i<unique.length; i+=BATCH_SIZE) {
      const batch = unique.slice(i,i+BATCH_SIZE);
      const batchPrompt = `分析情感，返回数组，长度${batch.length}，元素：{"valence":-1~1,"arousal":0~1,"emotion":"joy/sadness..."}。\n消息列表：${JSON.stringify(batch.map(m=>m.text))}`;
      let batchResult=[];
      for(let att=0;att<2;att++){
        try {
          const res = await callQWen([{ role:'user', content: batchPrompt }],0);
          let c = res.replace(/^```(?:json)?\s*\n?/i,'').replace(/\n?```\s*$/i,'');
          const arr = JSON.parse(c); if(Array.isArray(arr)){ batchResult=arr; break; }
        }catch(e){ if(att===1) batchResult=batch.map(m=>localSentiment(m.text)); }
      }
      while(batchResult.length<batch.length) batchResult.push({valence:0,arousal:0.5,emotion:'neutral'});
      for(let j=0;j<batch.length;j++) enriched.push({...batch[j], ...batchResult[j], date: batch[j].time || new Date().toISOString().slice(0,10)});
      processed+=batch.length;
      task.progress=25+Math.round(processed/total*75);
      task.message=`情感分析 ${processed}/${total}`;
    }
    task.status='completed'; task.progress=100; task.result=enriched; task.message='完成';
  } catch(e) { task.status='error'; task.error=e.message; }
}

// 导出给 Vercel
module.exports = app;

// 本地启动
if (require.main === module) {
  const PORT = process.env.PORT || 3001;
  app.listen(PORT, () => console.log(`✅ 本地运行: http://localhost:${PORT}`));
}