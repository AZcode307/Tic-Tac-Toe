import { useState, useEffect, useCallback, useRef } from "react";

// -- Q-Learning AI
________________________________________________
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
        const key = '${state}:${action}';
        return this.qTable.has(key) ? this.qTable.get(key) : 0.0;
    }

    setQ(state, action, value) {
        this.qTable.set('${state}:${action}', value);
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

// --- Storage helpers
________________________________________________