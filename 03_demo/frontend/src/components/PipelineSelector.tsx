import React from 'react'

interface Props {
  pipeline: 'yolo' | 'mediapipe'
  onChange: (pipeline: 'yolo' | 'mediapipe') => void
  disabled?: boolean
}

const pipelines: Array<{ id: 'yolo' | 'mediapipe'; name: string; description: string }> = [
  { id: 'yolo', name: 'YOLO', description: 'Детекция лица + классификация состояния глаз/зевоты' },
  { id: 'mediapipe', name: 'MediaPipe', description: 'Анализ EAR/MAR по ключевым точкам лица' },
]

export default function PipelineSelector({ pipeline, onChange, disabled }: Props) {
  return (
    <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
      <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">
        Конвейер
      </h2>
      <div className="grid grid-cols-2 gap-3">
        {pipelines.map((p) => {
          const active = pipeline === p.id
          return (
            <button
              key={p.id}
              onClick={() => !disabled && onChange(p.id)}
              disabled={disabled}
              className={[
                'rounded-lg border-2 p-3 text-left transition-all duration-150',
                active
                  ? 'border-blue-500 bg-blue-950 text-white'
                  : 'border-gray-700 bg-gray-800 text-gray-300 hover:border-gray-500',
                disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer',
              ].join(' ')}
            >
              <div className="font-semibold text-sm">{p.name}</div>
              <div className="text-xs text-gray-400 mt-1">{p.description}</div>
            </button>
          )
        })}
      </div>
    </div>
  )
}
