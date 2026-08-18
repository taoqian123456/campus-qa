const { createApp } = Vue;

const app = createApp({
  data() {
    return {
      // 相对地址：页面与 API 同源（http://127.0.0.1:8000/），file:// 打开会被浏览器拦 localStorage，已不支持
      apiBase: '',
      token: localStorage.getItem('qa_token') || '',
      username: localStorage.getItem('qa_username') || '',

      // 登录/注册
      authMode: 'login',
      authForm: { username: '', password: '', confirm: '' },
      authLoading: false,

      // 会话与消息
      sessions: [],
      currentSessionId: null,
      messages: [],
      inputText: '',
      sending: false,

      // 流式打字机状态
      streamSeq: 0,       // 递增使旧流失效
      streamController: null,  // 当前流的 AbortController（切换会话/退出时中止）
      typingTimer: null,
      typingBuf: '',
      pendingDone: null,
      typingMsg: null,
    };
  },
  computed: {
    loggedIn() { return !!this.token; },
  },
  mounted() {
    if (this.token) this.loadSessions();
  },

  methods: {
    /* ---------- 基础请求 ---------- */
    async api(path, opts = {}) {
      const headers = { 'Content-Type': 'application/json', ...(opts.headers || {}) };
      if (this.token) headers['Authorization'] = 'Bearer ' + this.token;
      let res;
      try {
        res = await fetch(this.apiBase + path, { ...opts, headers });
      } catch (e) {
        throw new Error('无法连接后端服务，请确认已启动 uvicorn');
      }
      if (res.status === 401) {
        if (this.token) { this.logout(); throw new Error('登录已过期，请重新登录'); }
        // 登录接口本身的 401 走下面的 detail 解析
      }
      if (!res.ok) {
        let detail = '请求失败（' + res.status + '）';
        try { const d = await res.json(); detail = d.detail || detail; } catch (e) {}
        throw new Error(detail);
      }
      return res;
    },

    /* ---------- 登录 / 注册 ---------- */
    async doAuth() {
      const { username, password, confirm } = this.authForm;
      if (!username || !password) { this.$message.warning('请输入用户名和密码'); return; }
      if (this.authMode === 'register') {
        if (password.length < 6) { this.$message.warning('密码至少 6 位'); return; }
        if (password !== confirm) { this.$message.warning('两次输入的密码不一致'); return; }
      }
      this.authLoading = true;
      try {
        const path = this.authMode === 'login' ? '/api/auth/login' : '/api/auth/register';
        const res = await this.api(path, {
          method: 'POST',
          body: JSON.stringify({ username, password }),
        });
        const data = await res.json();
        if (this.authMode === 'register') {
          this.$message.success('注册成功，请登录');
          this.authMode = 'login';
          this.authForm.password = '';
          this.authForm.confirm = '';
          return;
        }
        this.token = data.access_token;
        this.username = username;
        localStorage.setItem('qa_token', this.token);
        localStorage.setItem('qa_username', username);
        this.loadSessions();
      } catch (e) {
        this.$message.error(e.message);
      } finally {
        this.authLoading = false;
      }
    },

    logout() {
      this.invalidateStream();          // 中止当前流
      this.token = '';
      this.username = '';
      localStorage.removeItem('qa_token');
      localStorage.removeItem('qa_username');
      this.sessions = [];
      this.currentSessionId = null;
      this.messages = [];
      this.inputText = '';
    },

    /* ---------- 会话 ---------- */
    async loadSessions() {
      try {
        const res = await this.api('/api/chat/sessions');
        this.sessions = await res.json();
      } catch (e) {
        this.$message.error(e.message);
      }
    },

    newSession() {
      this.invalidateStream();          // 中止当前流
      this.currentSessionId = null;
      this.messages = [];
    },

    async switchSession(id) {
      if (id === this.currentSessionId) return;
      this.invalidateStream();          // 中止当前流
      this.currentSessionId = id;
      try {
        const res = await this.api(`/api/chat/sessions/${id}/messages`);
        const data = await res.json();
        this.messages = data.map(m => ({ role: m.role, content: m.content, sources: [], typing: false }));
        this.scrollToBottom();
      } catch (e) {
        this.$message.error(e.message);
      }
    },

    async deleteSession(id) {
      try {
        await this.$confirm('删除后聊天记录不可恢复，确定删除该会话？', '提示', { type: 'warning' });
      } catch (e) { return; } // 用户取消
      try {
        await this.api(`/api/chat/sessions/${id}`, { method: 'DELETE' });
        this.$message.success('已删除');
        if (this.currentSessionId === id) {
          this.currentSessionId = null;
          this.messages = [];
        }
        this.loadSessions();
      } catch (e) {
        this.$message.error(e.message);
      }
    },

    /* ---------- 发消息（SSE 流式 + 打字机） ---------- */
    async send() {
      const question = this.inputText.trim();
      if (!question || this.sending) return;

      let sessionId = this.currentSessionId;
      if (!sessionId) {
        try {
          const res = await this.api('/api/chat/sessions', {
            method: 'POST',
            body: JSON.stringify({ question }),
          });
          const data = await res.json();
          sessionId = data.id;
          this.currentSessionId = sessionId;
          this.loadSessions();
        } catch (e) {
          this.$message.error(e.message);
          return;
        }
      }

      this.inputText = '';
      this.messages.push({ role: 'user', content: question });
      // 关键：必须用 reactive() 包一层再存引用。若保存普通对象并在定时器里直接改它，
      // 修改会绕过 Vue 的响应式代理，界面不刷新——表现为一直"正在思考"，
      // 直到输入框打字触发一次无关的重新渲染才冒出整段答案。
      const aiMsg = Vue.reactive({ role: 'assistant', content: '', sources: [], typing: true });
      this.messages.push(aiMsg);
      this.sending = true;
      this.scrollToBottom();

      const seq = ++this.streamSeq;
      this.typingBuf = '';
      this.pendingDone = null;
      this.typingMsg = aiMsg;
      this.startTyping(aiMsg, seq);

      const controller = new AbortController();
      this.streamController = controller;
      try {
        await this.streamRequest(`/api/chat/sessions/${sessionId}/messages/stream`, { question }, seq, controller.signal);
      } catch (e) {
        if (seq === this.streamSeq) {
          this.finishStream(aiMsg, seq, e.message);  // 网络错误：清 loading + 兜底文案（Abort 不提示）
        }
      } finally {
        if (seq === this.streamSeq) {
          this.sending = false;
          this.streamController = null;
        }
      }
    },

    async streamRequest(path, body, seq, signal) {
      const headers = { 'Content-Type': 'application/json' };
      if (this.token) headers['Authorization'] = 'Bearer ' + this.token;
      let res;
      try {
        res = await fetch(this.apiBase + path, { method: 'POST', headers, body: JSON.stringify(body), signal });
      } catch (e) {
        if (e.name === 'AbortError') throw new Error('aborted');  // 主动中止：不报错、不覆盖消息
        throw new Error('无法连接后端服务，请确认已启动 uvicorn');
      }
      if (res.status === 401) { this.logout(); throw new Error('登录已过期，请重新登录'); }
      if (!res.ok) {
        let detail = '请求失败（' + res.status + '）';
        try { const d = await res.json(); detail = d.detail || detail; } catch (e) {}
        throw new Error(detail);
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buf = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        let idx;
        while ((idx = buf.indexOf('\n\n')) !== -1) {
          const frame = buf.slice(0, idx);
          buf = buf.slice(idx + 2);
          for (const line of frame.split('\n')) {
            if (line.startsWith('data: ')) {
              const ev = JSON.parse(line.slice(6));
              if (seq === this.streamSeq) this.onStreamEvent(ev);
            }
          }
        }
      }
      // 流结束（连接正常关闭）：缓冲里还有 token 或没等到 done 时，做兜底收尾
      if (seq !== this.streamSeq) throw new Error('aborted');
      if (buf.trim()) {
        for (const line of buf.split('\n')) {
          if (line.startsWith('data: ')) {
            const ev = JSON.parse(line.slice(6));
            if (seq === this.streamSeq) this.onStreamEvent(ev);
          }
        }
      }
      if (seq === this.streamSeq && !this.pendingDone && !this.typingMsg) {
        throw new Error('流已结束，但未收到完整回答');
      }
    },

    // 中止当前流：切换会话 / 退出登录 / 新流开始前调用
    invalidateStream() {
      this.streamSeq++;                    // 旧流事件全部失效
      if (this.streamController) this.streamController.abort();
      this.streamController = null;
      this.sending = false;
      clearInterval(this.typingTimer);
      this.typingTimer = null;
      this.typingBuf = '';
      this.pendingDone = null;
      this.typingMsg = null;
    },

    // 结束打字机：清 loading；传 errMsg 且尚无内容时显示兜底文案
    finishStream(msg, seq, errMsg) {
      if (!msg || seq !== this.streamSeq) return;
      msg.typing = false;
      if (errMsg && !msg.content) msg.content = '（' + errMsg + '）';
      if (!msg.sources) msg.sources = [];
      clearInterval(this.typingTimer);
      this.typingTimer = null;
      this.typingBuf = '';
      this.pendingDone = null;
      this.typingMsg = null;
      this.scrollToBottom();
    },

    onStreamEvent(ev) {
      if (ev.type === 'token') {
        this.typingBuf += ev.content;
      } else if (ev.type === 'done') {
        this.pendingDone = ev;
      } else if (ev.type === 'error') {
        this.finishStream(this.typingMsg, this.streamSeq, ev.message || '生成失败');
        this.$message.error(ev.message || '生成失败');
      }
    },

    startTyping(msg, seq) {
      clearInterval(this.typingTimer);
      this.typingTimer = setInterval(() => {
        if (seq !== this.streamSeq) { clearInterval(this.typingTimer); return; }
        if (this.typingBuf) {
          msg.content += this.typingBuf.slice(0, 3);   // 每次打 3 个字
          this.typingBuf = this.typingBuf.slice(3);
          this.scrollToBottom();
        } else if (this.pendingDone) {
          msg.content = this.pendingDone.answer;       // 收尾：与后端完整答案一致
          msg.sources = this.pendingDone.sources || [];
          this.finishStream(msg, seq, null);
        } else if (!msg.typing) {
          clearInterval(this.typingTimer);             // 出错后停止
        }
      }, 24);
    },

    scrollToBottom() {
      this.$nextTick(() => {
        const el = this.$refs.messagesEl;
        if (el) el.scrollTop = el.scrollHeight;
      });
    },
  },
});

app.use(ElementPlus, { locale: ElementPlusLocaleZhCn });
for (const [name, comp] of Object.entries(ElementPlusIconsVue)) {
  app.component(name, comp);
}
app.mount('#app');
