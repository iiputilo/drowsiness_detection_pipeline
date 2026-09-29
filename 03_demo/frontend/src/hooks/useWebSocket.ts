import { useState, useRef, useCallback } from 'react'

const WS_URL = 'ws://localhost:8000/stream'
const MAX_RECONNECT_ATTEMPTS = 5

export interface Metrics {
  ear: number | null
  mar: number | null
  gaze: number | null
  gaze_enabled?: boolean
  gaze_window_state?: {
    fraction: number
    alarm: boolean
    fill_ratio: number
    frame_count: number
  } | null
  inference_ms: number
  window_state: {
    fraction: number
    alarm: boolean
    fill_ratio: number
    frame_count: number
  }
  alarm_count: number
}

export interface WebSocketState {
  isConnected: boolean
  isStreaming: boolean
  isPaused: boolean
  currentState: 'alert' | 'drowsy' | 'distracted' | 'no_face' | 'idle'
  alarm: boolean
  metrics: Metrics | null
  mainFrame: string
  faceFrame: string | null
  alarmCount: number
}

export interface WebSocketActions {
  connect: (config: StreamConfig) => void
  disconnect: () => void
  sendControl: (action: string) => void
}

export interface StreamConfig {
  source: 'webcam' | 'video'
  pipeline: 'yolo' | 'mediapipe'
  show_overlay: boolean
  gaze_tracking?: boolean
  video_id?: string | null
}

const defaultState: WebSocketState = {
  isConnected: false,
  isStreaming: false,
  isPaused: false,
  currentState: 'idle',
  alarm: false,
  metrics: null,
  mainFrame: '',
  faceFrame: null,
  alarmCount: 0,
}

export function useWebSocket(): WebSocketState & WebSocketActions {
  const [state, setState] = useState<WebSocketState>(defaultState)
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectAttempts = useRef(0)
  const reconnectTimeout = useRef<ReturnType<typeof setTimeout> | null>(null)
  const pendingConfig = useRef<StreamConfig | null>(null)
  const intentionalClose = useRef(false)

  const disconnect = useCallback(() => {
    intentionalClose.current = true
    if (reconnectTimeout.current) {
      clearTimeout(reconnectTimeout.current)
      reconnectTimeout.current = null
    }
    if (wsRef.current) {
      try {
        wsRef.current.send(JSON.stringify({ action: 'stop' }))
      } catch (_) {}
      wsRef.current.close()
      wsRef.current = null
    }
    setState(defaultState)
    reconnectAttempts.current = 0
  }, [])

  const sendControl = useCallback((action: string) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action }))
      if (action === 'pause') {
        setState(prev => ({ ...prev, isPaused: true }))
      } else if (action === 'resume') {
        setState(prev => ({ ...prev, isPaused: false }))
      } else if (action === 'reset') {
        setState(prev => ({ ...prev, isPaused: false }))
      }
    }
  }, [])

  const connect = useCallback((config: StreamConfig) => {
    if (wsRef.current) {
      disconnect()
    }
    intentionalClose.current = false
    pendingConfig.current = config

    const ws = new WebSocket(WS_URL)
    wsRef.current = ws

    ws.onopen = () => {
      reconnectAttempts.current = 0
      setState(prev => ({ ...prev, isConnected: true, isStreaming: true }))
      if (pendingConfig.current) {
        ws.send(JSON.stringify(pendingConfig.current))
      }
    }

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)

        if (data.error) {
          console.error('Server error:', data.error)
          return
        }

        if (data.event === 'video_ended') {
          setState(prev => ({ ...prev, isStreaming: false, currentState: 'idle' }))
          return
        }

        setState(prev => ({
          ...prev,
          mainFrame: data.frame_main ?? prev.mainFrame,
          faceFrame: data.frame_face ?? null,
          currentState: data.state ?? prev.currentState,
          alarm: data.alarm ?? false,
          metrics: data.metrics ?? prev.metrics,
          alarmCount: data.metrics?.alarm_count ?? prev.alarmCount,
        }))
      } catch (e) {
        console.error('Failed to parse WebSocket message:', e)
      }
    }

    ws.onerror = () => {
      setState(prev => ({ ...prev, isConnected: false }))
    }

    ws.onclose = () => {
      wsRef.current = null
      setState(prev => ({ ...prev, isConnected: false, isStreaming: false }))

      if (
        !intentionalClose.current &&
        reconnectAttempts.current < MAX_RECONNECT_ATTEMPTS
      ) {
        const delay = Math.min(1000 * 2 ** reconnectAttempts.current, 30000)
        reconnectAttempts.current += 1
        reconnectTimeout.current = setTimeout(() => {
          if (pendingConfig.current) {
            connect(pendingConfig.current)
          }
        }, delay)
      }
    }
  }, [disconnect])

  return {
    ...state,
    connect,
    disconnect,
    sendControl,
  }
}
