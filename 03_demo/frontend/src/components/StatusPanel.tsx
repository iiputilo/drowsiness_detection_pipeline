import React from 'react'
import { Metrics } from '../hooks/useWebSocket'

interface Props {
  currentState: 'alert' | 'drowsy' | 'distracted' | 'no_face' | 'idle'
  alarm: boolean
  metrics: Metrics | null
  pipeline: 'yolo' | 'mediapipe'
  alarmCount: number
}

const EAR_THRESHOLD = 0.25
const MAR_THRESHOLD = 0.6
const GAZE_THRESHOLD = 0.35

export default function StatusPanel({ currentState, alarm, metrics, pipeline, alarmCount }: Props) {
  const stateBg: Record<string, string> = {
    alert: 'bg-green-900',
    drowsy: 'bg-red-900',
    distracted: 'bg-amber-900',
    no_face: 'bg-gray-700',
    idle: 'bg-gray-800',
  }

  const stateLabel: Record<string, string> = {
    alert: 'БОДРСТВУЕТ',
    drowsy: 'ЗАСЫПАЕТ',
    distracted: 'ОТВЛЕЧЕН',
    no_face: 'НЕТ ЛИЦА',
    idle: 'ОЖИДАНИЕ',
  }

  const WINDOW_DURATION = 2.0
  const DROWSY_FRACTION = 0.8
  const windowState = metrics?.window_state
  const gazeWindowState = metrics?.gaze_window_state
  const fraction = windowState?.fraction ?? 0
  const fillRatio = windowState?.fill_ratio ?? 0
  const frameCount = windowState?.frame_count ?? 0
  const windowFull = fillRatio >= 1.0

  const gazeFraction = gazeWindowState?.fraction ?? 0
  const gazeFillRatio = gazeWindowState?.fill_ratio ?? 0
  const gazeFrameCount = gazeWindowState?.frame_count ?? 0
  const gazeWindowFull = gazeFillRatio >= 1.0

  return (
    <div className="bg-gray-900 rounded-xl border border-gray-800 p-4 flex flex-col gap-3">
      <div
        className={[
          'w-full h-20 rounded-lg flex items-center justify-center',
          stateBg[currentState] ?? 'bg-gray-800',
          alarm ? 'animate-pulse' : '',
        ].join(' ')}
      >
        <span className="text-2xl font-bold text-white">
          {stateLabel[currentState] ?? currentState.toUpperCase()}
        </span>
      </div>

      <div>
        <div className="flex justify-between text-xs text-gray-500 mb-1.5">
          <span>Окно детекции ({WINDOW_DURATION}с)</span>
          <span className={`font-mono ${windowFull && fraction >= DROWSY_FRACTION ? 'text-red-400' : 'text-gray-400'}`}>
            {frameCount > 0 ? `${(fraction * 100).toFixed(0)}%` : '—'}
          </span>
        </div>

        {/* fraction bar */}
        <div className="relative h-3 bg-gray-700 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-100 ${
              windowFull && fraction >= DROWSY_FRACTION ? 'bg-red-500' : 'bg-blue-500'
            } ${!windowFull ? 'opacity-40' : ''}`}
            style={{ width: `${Math.min(fraction * 100, 100)}%` }}
          />
        </div>

        {/* threshold marker — rendered outside overflow-hidden parent */}
        <div className="relative h-0">
          <div
            className="absolute -top-3 w-px h-3 bg-white/50"
            style={{ left: `${DROWSY_FRACTION * 100}%` }}
          />
        </div>

        <div className="flex justify-between text-xs mt-1.5">
          <span className="text-gray-600">
            {windowFull ? `${frameCount} кадров` : `прогрев ${(fillRatio * 100).toFixed(0)}%…`}
          </span>
          <span className="text-gray-600">порог {(DROWSY_FRACTION * 100).toFixed(0)}%</span>
        </div>
      </div>

      {pipeline === 'mediapipe' && metrics?.gaze_enabled && gazeWindowState && (
        <div>
          <div className="flex justify-between text-xs text-gray-500 mb-1.5">
            <span>Окно детекции (взгляд)</span>
            <span className={`font-mono ${gazeWindowFull && gazeFraction >= DROWSY_FRACTION ? 'text-amber-400' : 'text-gray-400'}`}>
              {gazeFrameCount > 0 ? `${(gazeFraction * 100).toFixed(0)}%` : '—'}
            </span>
          </div>

          <div className="relative h-3 bg-gray-700 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-100 ${
                gazeWindowFull && gazeFraction >= DROWSY_FRACTION ? 'bg-amber-500' : 'bg-blue-500'
              } ${!gazeWindowFull ? 'opacity-40' : ''}`}
              style={{ width: `${Math.min(gazeFraction * 100, 100)}%` }}
            />
          </div>

          <div className="relative h-0">
            <div
              className="absolute -top-3 w-px h-3 bg-white/50"
              style={{ left: `${DROWSY_FRACTION * 100}%` }}
            />
          </div>

          <div className="flex justify-between text-xs mt-1.5">
            <span className="text-gray-600">
              {gazeWindowFull ? `${gazeFrameCount} кадров` : `прогрев ${(gazeFillRatio * 100).toFixed(0)}%…`}
            </span>
            <span className="text-gray-600">порог {(DROWSY_FRACTION * 100).toFixed(0)}%</span>
          </div>
        </div>
      )}

      {pipeline === 'mediapipe' && metrics && (
        <div className="space-y-1">
          <div className="text-xs text-gray-500">Метрики</div>
          <div className="flex justify-between text-sm">
            <span className="text-gray-400">EAR:</span>
            <span
              className={
                metrics.ear !== null && metrics.ear < EAR_THRESHOLD
                  ? 'text-red-400 font-mono'
                  : 'text-green-400 font-mono'
              }
            >
              {metrics.ear !== null ? metrics.ear.toFixed(3) : 'N/A'}
            </span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-gray-400">MAR:</span>
            <span
              className={
                metrics.mar !== null && metrics.mar > MAR_THRESHOLD
                  ? 'text-red-400 font-mono'
                  : 'text-green-400 font-mono'
              }
            >
              {metrics.mar !== null ? metrics.mar.toFixed(3) : 'N/A'}
            </span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-gray-400">GAZE:</span>
            <span
              className={
                metrics.gaze !== null && metrics.gaze > GAZE_THRESHOLD
                  ? 'text-amber-400 font-mono'
                  : 'text-green-400 font-mono'
              }
            >
              {metrics.gaze !== null ? metrics.gaze.toFixed(3) : 'N/A'}
            </span>
          </div>
        </div>
      )}

      <div className="flex justify-between text-xs text-gray-500 mt-auto">
        <span>
          Инференс:{' '}
          <span className="font-mono text-gray-400">
            {metrics ? `${metrics.inference_ms.toFixed(1)}мс` : '--'}
          </span>
        </span>
        <span>
          Тревоги:{' '}
          <span className="font-mono text-gray-400">{alarmCount}</span>
        </span>
      </div>
    </div>
  )
}
