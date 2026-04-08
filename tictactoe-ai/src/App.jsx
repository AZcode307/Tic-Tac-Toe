import { useState, useEffect, useCallback, useRef } from "react";

// -- Q-Learning AI ________________________________________________
const ALPHA = 0.3; //learning rate
const GAMMA = 0.9; // discount factor
const EPSILON_START = 0.4; // initial exploration rate
const EPSILON_MIN = 0.05; //minimum exploration

function boardKey(board) {
    return board.map(c => c || "-").join("");
}

function getWinner(board) {
    const lines = [
        [0,1,2], [3,4,5], [6,7,8],
        [0,3,6], [1,4,7], [2,5,8],
        [0,4,8], [2,4,6]
    ];
    for (const [a,b,c] of lines) {
        if(board[a] && board[a] === board[b] && board[a] == board[c]) return board[a];
    }
    return null;
}

function isDraw(board) {
    return board.every(c => c != null) && !getWinner(board);
}

function getEmptySquares(board) {
    return board.map((c,i) => c === null ? i : -1).filter(i => i !== -1);
}

class QLearningAgent {
    constructor(savedState = null) {
        if (savedState) {
            this.qTable = new Map(Object.entries(savedState.qTable || {}));
            this.gamesPlayed = savedState.gamesPlayed || 0;
            this.wins = savedState.wins || 0;
            this.losses = savedState.losses || 0;
            this.draws = savedState.draws || 0;
            this.playerMoveFreq = savedState.playerMoveFreq || {};
        }
        else{
            this.qTable = new Map();
            this.gamesPlayed = 0;
            this.wins = 0;
            this.losses = 0;
            this.draws = 0;
            this.playerMoveFreq = {}; //track opponent patterns
        }
        this.episodeHistory = []; // (state, action) paris for current game
    }
    getEpsilon() {
        // Decay exploration as moe games are played
        const decay = Math.max(EPSILON_MIN, EPSILON_START - (this.gamesPlayed * 0.005));
        return decay;
    }

    getQ(state, action) {
        const key = `${state}:${action}`;
        return this.qTable.has(key) ? this.qTable.get(key) : 0.0;
    }

    setQ(state, action, value) {
        this.qTable.set(`${state}:${action}`, value);
    }

    chooseAction(board) {
        const state = boardKey(board);
        const available = getEmptySquares(board);
        if (available.length === 0) return null;
        // Epsilon-greedy
        if (Math.random() < this.getEpsilon()){
            // Exploration: sometimes bias toward countering player's favorate spots
            return this._explorationMove(board, available);
        }
        // Exploitation: pick highest Q
        let best = -Infinity;
        let bestAction = available[0];
        for (const a of available) {
            const q = this.getQ(state, a);
            if (q > best) { best = q; bestAction = a; }
        }
        return bestAction;
    }
    _explorationMove(board, available) {
        // With some probablity, block players favorite square
        const freqKeys = Object.keys(this.playerMoveFreq);
        if (freqKeys.length > 0 && Math.random() < 0.4) {
            const sorted = freqKeys
                .map(k => ({ sq: parseInt(k), freq: this.playerMoveFreq[k]}))
                .sort((a,b) => b.freq - a.freq);
            for (const { sq } of sorted) {
                if (available.includes(sq)) return sq;
            }
        }
        return available[Math.floor(Math.random() * available.length)];
    }

    recordMove(board, action) {
        this.episodeHistory.push({ state: boardKey(board), action});
    }
    recordPlayerMove(action) {
        this.playerMoveFreq[action] = (this.playerMoveFreq[action] || 0) + 1;
    }

    endGame(result) {
        // result: 1 = AI win, -1 = AI loss, 0 = draw
        const rewards = { 1: 1.0, "-1": -1.0, 0: 0.1 };
        const finalReward = rewards[result] ?? 0;

        this.gamesPlayed++;
        if (result === 1) this.wins++;
        else if (result === -1) this.losses++;
        else this.draws++;

        // Backpropagate rewards through episode
        let nextMaxQ = 0;
        for (let i = this.episodeHistory.length - 1; i >= 0; i--) {
            const { state, action } = this.episodeHistory[i];
            const reward = i === this.episodeHistory.length - 1 ? finalReward : 0;
            const oldQ = this.getQ(state, action);
            const newQ = oldQ + ALPHA * (reward + GAMMA * nextMaxQ - oldQ);
            this.setQ(state, action, newQ);
            nextMaxQ = newQ;
        }

        this.episodeHistory = [];
    }
    
    serialize() {
        return {
            qTable: Object.fromEntries(this.qTable),
            gamesPlayed: this.gamesPlayed,
            wins: this.wins,
            losses: this.losses,
            draws: this.draws,
            playerMoveFreq: this.playerMoveFreq,
        };
    }

    getTopPlayerSquares() {
        return Object.entries(this.playerMoveFreq)
            .sort(([,a],[,b]) => b - a)
            .slice(0, 3)
            .map(([sq]) => parseInt(sq));
    }
}

// --- Storage helpers ________________________________________________

const STORAGE_KEY = "ttt_ai_state";

async function loadAI() {
    try {
        const raw = localStorage.getItem(STORAGE_KEY)
        if (raw) return new QLearningAgent(JSON.parse(raw));
    } catch (_) {}
    return new QLearningAgent();
}

async function saveAI(agent) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(agent.serialize()));
}

// ---- UI Components ______________________________________________________

const SQUARE_LABELS = ["TL", "TC", "TR","ML","MC","MR","BL","BC","BR"];

const LINES = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]];

function checkWinner(b) {
  for (const [a,bb,c] of LINES) {
    if (b[a] && b[a] === b[bb] && b[a] === b[c]) return { winner: b[a], line: [a,bb,c] };
  }
  if (b.every(c => c !== null)) return { winner: "draw", line: [] };
  return null;
}

function Square({ value, index, onClick, highlight, aiTarget, playerFav }) {
    const isEmpty = value === null;
    return( 
    <button
    onClick={onClick}
    style={{
        width: 96, height: 96,
        background: highlight
          ? "rgba(255,220,50,0.18)"
          : aiTarget
          ? "rgba(255,80,80,0.10)"
          : playerFav
          ? "rgba(80,180,255,0.10)"
          : "rgba(255,255,255,0.04)",
        border: highlight
          ? "2px solid rgba(255,220,50,0.7)"
          : "2px solid rgba(255,255,255,0.10)",
        borderRadius: 16,
        cursor: isEmpty ? "pointer" : "default",
        fontSize: 42,
        fontFamily: "'Courier Prime', monospace",
        fontWeight: 700,
        color: value === "X" ? "#60c8ff" : value === "O" ? "#ff6b6b" : "transparent",
        transition: "all 0.18s ease",
        boxShadow: highlight ? "0 0 24px rgba(255,220,50,0.3)" : "none",
        transform: highlight ? "scale(1.06)" : "scale(1)",
        position: "relative",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
      }}
      onMouseEnter={e => { if (isEmpty) e.currentTarget.style.background = "rgba(255,255,255,0.09)"; }}
      onMouseLeave={e => { if (isEmpty) e.currentTarget.style.background = highlight ? "rgba(255,220,50,0.18)" : aiTarget ? "rgba(255,80,80,0.10)" : playerFav ? "rgba(80,180,255,0.10)" : "rgba(255,255,255,0.04)"; }}
    >
      {value}
      {!value && aiTarget && (
        <span style={{ position:"absolute", bottom:4, right:6, fontSize:9, color:"rgba(255,80,80,0.5)", fontFamily:"monospace" }}>AI↑</span>
      )}
      {!value && playerFav && (
        <span style={{ position:"absolute", bottom:4, right:6, fontSize:9, color:"rgba(80,180,255,0.5)", fontFamily:"monospace" }}>★</span>
      )}
    </button>
  );
}

function StatBar({ label, value, total, color }) {
  const pct = total > 0 ? Math.round((value / total) * 100) : 0;
  return (
    <div style={{ marginBottom: 8 }}>
      <div style={{ display:"flex", justifyContent:"space-between", fontSize:12, color:"rgba(255,255,255,0.6)", marginBottom:3 }}>
        <span>{label}</span>
        <span>{value} <span style={{opacity:.5}}>({pct}%)</span></span>
      </div>
      <div style={{ height:5, background:"rgba(255,255,255,0.08)", borderRadius:3, overflow:"hidden" }}>
        <div style={{ height:"100%", width:`${pct}%`, background:color, borderRadius:3, transition:"width 0.4s ease" }} />
      </div>
    </div>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function TicTacToeAI() {
  const [board, setBoard] = useState(Array(9).fill(null));
  const [playerTurn, setPlayerTurn] = useState(true); // player = X, AI = O
  const [winner, setWinner] = useState(null); // "X" | "O" | "draw" | null
  const [winLine, setWinLine] = useState([]);
  const [status, setStatus] = useState("Your turn");
  const [agent, setAgent] = useState(null);
  const [stats, setStats] = useState({ gamesPlayed:0, wins:0, losses:0, draws:0 });
  const [topSquares, setTopSquares] = useState([]);
  const [aiWatching, setAiWatching] = useState([]);
  const [loaded, setLoaded] = useState(false);
  // eslint-disable-next-line no-unused-vars
  const [lastResult, setLastResult] = useState(null);
  const agentRef = useRef(null);

  // Load agent from storage
  useEffect(() => {
    loadAI().then(a => {
      agentRef.current = a;
      setAgent(a);
      setStats({ gamesPlayed:a.gamesPlayed, wins:a.wins, losses:a.losses, draws:a.draws });
      setTopSquares(a.getTopPlayerSquares());
      setLoaded(true);
    });
  }, []);

  const doAiMove = useCallback((currentBoard, currentAgent) => {
    setTimeout(() => {
      const action = currentAgent.chooseAction(currentBoard);
      if (action === null) return;

      currentAgent.recordMove(currentBoard, action);

      const newBoard = [...currentBoard];
      newBoard[action] = "O";
      setBoard(newBoard);

      const result = checkWinner(newBoard);
      if (result) {
        if (result.winner === "O") {
          currentAgent.endGame(1);
          setStatus("AI wins! 🤖");
          setLastResult("loss");
        } else if (result.winner === "draw") {
          currentAgent.endGame(0);
          setStatus("Draw! 🤝");
          setLastResult("draw");
        }
        setWinner(result.winner);
        setWinLine(result.line);
        setStats({ gamesPlayed:currentAgent.gamesPlayed, wins:currentAgent.wins, losses:currentAgent.losses, draws:currentAgent.draws });
        setTopSquares(currentAgent.getTopPlayerSquares());
        saveAI(currentAgent);
      } else {
        setPlayerTurn(true);
        setStatus("Your turn");
        // Show AI's watched squares
        setAiWatching(currentAgent.getTopPlayerSquares());
      }
      setAgent({...currentAgent});
    }, 480);
  }, []);

  function handleSquareClick(i) {
    if (!loaded || !playerTurn || board[i] || winner) return;
    const a = agentRef.current;

    a.recordPlayerMove(i);

    const newBoard = [...board];
    newBoard[i] = "X";
    setBoard(newBoard);
    setPlayerTurn(false);
    setStatus("AI thinking…");
    setAiWatching([]);

    const result = checkWinner(newBoard);
    if (result) {
      if (result.winner === "X") {
        a.endGame(-1);
        setStatus("You win! 🎉");
        setLastResult("win");
      } else if (result.winner === "draw") {
        a.endGame(0);
        setStatus("Draw! 🤝");
        setLastResult("draw");
      }
      setWinner(result.winner);
      setWinLine(result.line);
      setStats({ gamesPlayed:a.gamesPlayed, wins:a.wins, losses:a.losses, draws:a.draws });
      setTopSquares(a.getTopPlayerSquares());
      saveAI(a);
      setAgent({...a});
      return;
    }

    doAiMove(newBoard, a);
  }

  function resetGame() {
    setBoard(Array(9).fill(null));
    setPlayerTurn(true);
    setWinner(null);
    setWinLine([]);
    setStatus("Your turn");
    setLastResult(null);
    setAiWatching(agentRef.current?.getTopPlayerSquares() || []);
  }

  async function resetAI() {
    const fresh = new QLearningAgent();
    agentRef.current = fresh;
    await saveAI(fresh);
    setAgent(fresh);
    setStats({ gamesPlayed:0, wins:0, losses:0, draws:0 });
    setTopSquares([]);
    setAiWatching([]);
    resetGame();
  }

  const epsilonPct = agent ? Math.round(Math.max(EPSILON_MIN, EPSILON_START - ((agent.gamesPlayed || 0) * 0.005)) * 100) : Math.round(EPSILON_START * 100);
  const qTableSize = agent?.qTable?.size || agentRef.current?.qTable?.size || 0;

  return (
    <div style={{
      minHeight: "100vh",
      background: "#0d0d14",
      backgroundImage: "radial-gradient(ellipse at 20% 50%, rgba(30,20,60,0.8) 0%, transparent 60%), radial-gradient(ellipse at 80% 20%, rgba(10,30,50,0.7) 0%, transparent 50%)",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      fontFamily: "'Courier Prime', 'Courier New', monospace",
      padding: 20,
      gap: 32,
      flexWrap: "wrap",
    }}>
      {/* Left panel */}
      <div style={{ width: 220 }}>
        <div style={{
          background: "rgba(255,255,255,0.04)",
          border: "1px solid rgba(255,255,255,0.08)",
          borderRadius: 16,
          padding: "20px 18px",
          marginBottom: 16,
        }}>
          <div style={{ fontSize: 11, letterSpacing: 3, color: "rgba(255,255,255,0.35)", marginBottom: 14, textTransform:"uppercase" }}>Record</div>
          <StatBar label="Your wins" value={stats.losses} total={stats.gamesPlayed} color="#60c8ff" />
          <StatBar label="AI wins" value={stats.wins} total={stats.gamesPlayed} color="#ff6b6b" />
          <StatBar label="Draws" value={stats.draws} total={stats.gamesPlayed} color="#aaa" />
          <div style={{ marginTop:12, fontSize:11, color:"rgba(255,255,255,0.3)", textAlign:"center" }}>
            {stats.gamesPlayed} games played
          </div>
        </div>

        <div style={{
          background: "rgba(255,255,255,0.04)",
          border: "1px solid rgba(255,255,255,0.08)",
          borderRadius: 16,
          padding: "20px 18px",
        }}>
          <div style={{ fontSize: 11, letterSpacing: 3, color: "rgba(255,255,255,0.35)", marginBottom: 14, textTransform:"uppercase" }}>AI Brain</div>

          <div style={{ display:"flex", justifyContent:"space-between", fontSize:12, color:"rgba(255,255,255,0.5)", marginBottom:8 }}>
            <span>Explore rate</span>
            <span style={{color: epsilonPct > 20 ? "#ffa94d" : "#a9e34b"}}>{epsilonPct}%</span>
          </div>
          <div style={{ height:4, background:"rgba(255,255,255,0.08)", borderRadius:2, marginBottom:16 }}>
            <div style={{ height:"100%", width:`${epsilonPct}%`, background: epsilonPct > 20 ? "#ffa94d" : "#a9e34b", borderRadius:2, transition:"width .4s" }} />
          </div>

          <div style={{ fontSize:12, color:"rgba(255,255,255,0.5)", marginBottom:4 }}>Q-table entries</div>
          <div style={{ fontSize:22, fontWeight:700, color:"#e0e0e0", marginBottom:16 }}>{qTableSize.toLocaleString()}</div>

          {topSquares.length > 0 && (
            <>
              <div style={{ fontSize:11, color:"rgba(255,255,255,0.35)", marginBottom:8, letterSpacing:2, textTransform:"uppercase" }}>Your patterns</div>
              <div style={{ fontSize:12, color:"rgba(80,180,255,0.8)" }}>
                You often play: {topSquares.map(s => SQUARE_LABELS[s]).join(", ")}
              </div>
            </>
          )}
        </div>
      </div>

      {/* Center: board */}
      <div style={{ display:"flex", flexDirection:"column", alignItems:"center", gap: 20 }}>
        <div>
          <h1 style={{
            margin: 0,
            fontSize: 28,
            fontWeight: 700,
            color: "#fff",
            letterSpacing: 4,
            textAlign: "center",
            textTransform: "uppercase",
          }}>
            Tic·Tac·Toe
          </h1>
          <div style={{ textAlign:"center", fontSize:12, color:"rgba(255,255,255,0.3)", letterSpacing:3, marginTop:4 }}>
            ADAPTIVE AI
          </div>
        </div>

        {/* Status */}
        <div style={{
          fontSize: 15,
          color: winner === "X" ? "#60c8ff" : winner === "O" ? "#ff6b6b" : winner === "draw" ? "#aaa" : "rgba(255,255,255,0.7)",
          fontWeight: 600,
          letterSpacing: 1,
          height: 24,
          transition: "color 0.3s",
        }}>
          {status}
        </div>

        {/* Board */}
        <div style={{
          display: "grid",
          gridTemplateColumns: "repeat(3, 96px)",
          gap: 10,
          padding: 20,
          background: "rgba(255,255,255,0.03)",
          border: "1px solid rgba(255,255,255,0.07)",
          borderRadius: 20,
        }}>
          {board.map((val, i) => (
            <Square
              key={i}
              value={val}
              index={i}
              onClick={() => handleSquareClick(i)}
              highlight={winLine.includes(i)}
              aiTarget={!val && aiWatching.includes(i) && playerTurn && !winner}
              playerFav={!val && topSquares.includes(i) && !aiWatching.includes(i) && playerTurn && !winner}
            />
          ))}
        </div>

        {/* Legend */}
        <div style={{ display:"flex", gap:20, fontSize:11, color:"rgba(255,255,255,0.3)" }}>
          <span>🔵 = You (X)</span>
          <span>🔴 = AI (O)</span>
        </div>
        {!winner && playerTurn && topSquares.length > 0 && (
          <div style={{ fontSize:11, color:"rgba(255,255,255,0.25)", textAlign:"center" }}>
            <span style={{color:"rgba(255,80,80,0.5)"}}>AI↑</span> = AI watching · <span style={{color:"rgba(80,180,255,0.5)"}}>★</span> = your fav square
          </div>
        )}

        {/* Buttons */}
        <div style={{ display:"flex", gap:10 }}>
          <button onClick={resetGame} style={{
            padding: "10px 24px",
            background: winner ? "rgba(96,200,255,0.15)" : "rgba(255,255,255,0.07)",
            border: `1px solid ${winner ? "rgba(96,200,255,0.4)" : "rgba(255,255,255,0.12)"}`,
            borderRadius: 10,
            color: winner ? "#60c8ff" : "rgba(255,255,255,0.6)",
            cursor: "pointer",
            fontSize: 13,
            fontFamily: "inherit",
            fontWeight: 600,
            letterSpacing: 1,
            transition: "all 0.2s",
          }}>
            {winner ? "Play Again" : "Reset"}
          </button>
          <button onClick={resetAI} style={{
            padding: "10px 16px",
            background: "rgba(255,80,80,0.07)",
            border: "1px solid rgba(255,80,80,0.2)",
            borderRadius: 10,
            color: "rgba(255,100,100,0.6)",
            cursor: "pointer",
            fontSize: 12,
            fontFamily: "inherit",
            letterSpacing: 1,
            transition: "all 0.2s",
          }}>
            Reset AI
          </button>
        </div>
      </div>

      {/* Right: how it works */}
      <div style={{ width: 210 }}>
        <div style={{
          background: "rgba(255,255,255,0.04)",
          border: "1px solid rgba(255,255,255,0.08)",
          borderRadius: 16,
          padding: "20px 18px",
        }}>
          <div style={{ fontSize:11, letterSpacing:3, color:"rgba(255,255,255,0.35)", marginBottom:14, textTransform:"uppercase" }}>How It Learns</div>
          {[
            { icon:"🧠", title:"Q-Learning", desc:"Maps board states → move values, updated after every game." },
            { icon:"👁️", title:"Pattern Memory", desc:"Tracks your favorite squares and tries to counter them." },
            { icon:"📉", title:"Exploration Decay", desc:"Starts random, becomes strategic as games increase." },
            { icon:"⚡", title:"Persistent", desc:"Memory survives page reloads — it never forgets you." },
          ].map(({ icon, title, desc }) => (
            <div key={title} style={{ marginBottom:14 }}>
              <div style={{ fontSize:13, color:"rgba(255,255,255,0.75)", fontWeight:600, marginBottom:3 }}>{icon} {title}</div>
              <div style={{ fontSize:11, color:"rgba(255,255,255,0.35)", lineHeight:1.5 }}>{desc}</div>
            </div>
          ))}
          <div style={{ marginTop:8, padding:"10px 12px", background:"rgba(255,220,50,0.06)", border:"1px solid rgba(255,220,50,0.15)", borderRadius:8, fontSize:11, color:"rgba(255,220,50,0.6)", lineHeight:1.5 }}>
            💡 Play ~20+ games to see the AI adapt to your style
          </div>
        </div>
      </div>
    </div>
  );
}