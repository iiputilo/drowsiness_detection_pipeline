import React from 'react'
import { Camera } from 'lucide-react'

interface Props {
  pipeline: 'yolo' | 'mediapipe'
  mainFrame: string
  faceFrame: string | null
  isStreaming: boolean
}

export default function VideoDisplay({ pipeline, mainFrame, faceFrame, isStreaming }: Props) {
  if (!isStreaming || !mainFrame) {
    return (
      <div className="bg-gray-900 rounded-xl border border-gray-800 flex items-center justify-center h-full min-h-72">
        <div className="flex flex-col items-center gap-3 text-gray-500">
          <Camera size={48} />
          <span className="text-sm">Выберите источник и нажмите Старт</span>
        </div>
      </div>
    )
  }

  if (pipeline === 'mediapipe') {
    return (
      <div className="bg-gray-900 rounded-xl border border-gray-800 overflow-hidden">
        <img
          src={`data:image/jpeg;base64,${mainFrame}`}
          alt="Video stream"
          className="w-full h-auto object-contain"
        />
      </div>
    )
  }

  return (
    <div className="bg-gray-900 rounded-xl border border-gray-800 overflow-hidden">
      <div className="grid grid-cols-2 gap-0">
        <div className="overflow-hidden">
          <img
            src={`data:image/jpeg;base64,${mainFrame}`}
            alt="Полный кадр"
            className="w-full h-auto object-contain"
          />
          <div className="text-center text-xs text-gray-500 py-1">Полный кадр</div>
        </div>
        <div className="overflow-hidden border-l border-gray-800">
          {faceFrame ? (
            <>
              <img
                src={`data:image/jpeg;base64,${faceFrame}`}
                alt="Кроп лица"
                className="w-full h-auto object-contain"
              />
              <div className="text-center text-xs text-gray-500 py-1">Кроп лица</div>
            </>
          ) : (
            <div className="flex items-center justify-center h-full min-h-48 text-gray-600 text-sm">
              Лицо не обнаружено
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
