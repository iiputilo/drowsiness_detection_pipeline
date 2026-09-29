import React, { useRef } from 'react'
import { Play, Pause, RotateCcw, Upload } from 'lucide-react'

interface Props {
  source: 'webcam' | 'video'
  onSourceChange: (s: 'webcam' | 'video') => void
  pipeline: 'yolo' | 'mediapipe'
  showOverlay: boolean
  onShowOverlayChange: (v: boolean) => void
  gazeTracking: boolean
  onGazeTrackingChange: (v: boolean) => void
  isConnected: boolean
  isStreaming: boolean
  isPaused: boolean
  uploadedVideoId: string | null
  onVideoUploaded: (id: string) => void
  onStart: () => void
  onStop: () => void
  onPause: () => void
  onResume: () => void
  onReset: () => void
}

export default function ControlPanel({
  source, onSourceChange,
  pipeline,
  showOverlay, onShowOverlayChange,
  gazeTracking, onGazeTrackingChange,
  isConnected, isStreaming, isPaused,
  uploadedVideoId, onVideoUploaded,
  onStart, onStop, onPause, onResume, onReset,
}: Props) {
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileDrop = async (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    const file = e.dataTransfer.files[0]
    if (file) await uploadFile(file)
  }

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) await uploadFile(file)
  }

  const uploadFile = async (file: File) => {
    try {
      const formData = new FormData()
      formData.append('file', file)
      const res = await fetch('http://localhost:8000/upload_video', {
        method: 'POST',
        body: formData,
      })
      const data = await res.json()
      if (data.video_id) {
        onVideoUploaded(data.video_id)
      }
    } catch (e) {
      console.error('Upload failed:', e)
    }
  }

  return (
    <div className="bg-gray-900 rounded-xl border border-gray-800 p-4 relative">
      <div className="absolute top-4 right-4 flex items-center gap-1.5">
        <div className={`w-2.5 h-2.5 rounded-full ${isConnected ? 'bg-green-500' : 'bg-gray-600'}`} />
        <span className="text-xs text-gray-500">{isConnected ? 'Подключено' : 'Отключено'}</span>
      </div>

      <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">Управление</h2>

      <div className="mb-3">
        <div className="text-xs text-gray-500 mb-1.5">Источник</div>
        <div className="flex gap-2">
          {(['webcam', 'video'] as const).map((s) => (
            <button
              key={s}
              onClick={() => !isStreaming && onSourceChange(s)}
              disabled={isStreaming}
              className={[
                'flex-1 py-1.5 px-3 rounded-lg text-sm font-medium transition-colors',
                source === s
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-800 text-gray-400 hover:bg-gray-700',
                isStreaming ? 'opacity-50 cursor-not-allowed' : '',
              ].join(' ')}
            >
              {s === 'webcam' ? 'Веб-камера' : 'Видеофайл'}
            </button>
          ))}
        </div>
      </div>

      {source === 'video' && (
        <div className="mb-3">
          <div
            className={[
              'border-2 border-dashed rounded-lg p-4 text-center cursor-pointer transition-colors',
              uploadedVideoId ? 'border-green-600 bg-green-950' : 'border-gray-700 hover:border-gray-500',
            ].join(' ')}
            onDrop={handleFileDrop}
            onDragOver={(e) => e.preventDefault()}
            onClick={() => fileInputRef.current?.click()}
          >
            <Upload size={20} className="mx-auto mb-1 text-gray-500" />
            {uploadedVideoId ? (
              <p className="text-xs text-green-400">Видео загружено</p>
            ) : (
              <p className="text-xs text-gray-500">Перетащите файл или нажмите для выбора</p>
            )}
            <input
              ref={fileInputRef}
              type="file"
              accept="video/*"
              className="hidden"
              onChange={handleFileSelect}
            />
          </div>

          {isStreaming && (
            <div className="flex gap-2 mt-2">
              <button
                onClick={onReset}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-gray-800 text-gray-300 hover:bg-gray-700 text-sm"
              >
                <RotateCcw size={14} />
                Сбросить
              </button>
              <button
                onClick={isPaused ? onResume : onPause}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-gray-800 text-gray-300 hover:bg-gray-700 text-sm"
              >
                {isPaused ? <Play size={14} /> : <Pause size={14} />}
                {isPaused ? 'Продолжить' : 'Пауза'}
              </button>
            </div>
          )}
        </div>
      )}

      <div className="flex items-center justify-between mb-4">
        <span className="text-sm text-gray-300">Наложение</span>
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-500">{showOverlay ? 'Вкл' : 'Выкл'}</span>
          <button
            onClick={() => onShowOverlayChange(!showOverlay)}
            className={[
              'relative inline-flex h-5 w-9 items-center rounded-full transition-colors',
              showOverlay ? 'bg-blue-600' : 'bg-gray-700',
            ].join(' ')}
          >
            <span
              className={[
                'inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform',
                showOverlay ? 'translate-x-4' : 'translate-x-1',
              ].join(' ')}
            />
          </button>
        </div>
      </div>

      {pipeline === 'mediapipe' && (
        <div className="flex items-center justify-between mb-4">
          <span className="text-sm text-gray-300">Трекинг взгляда</span>
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-500">{gazeTracking ? 'Вкл' : 'Выкл'}</span>
            <button
              onClick={() => !isStreaming && onGazeTrackingChange(!gazeTracking)}
              disabled={isStreaming}
              className={[
                'relative inline-flex h-5 w-9 items-center rounded-full transition-colors',
                gazeTracking ? 'bg-blue-600' : 'bg-gray-700',
                isStreaming ? 'opacity-50 cursor-not-allowed' : '',
              ].join(' ')}
            >
              <span
                className={[
                  'inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform',
                  gazeTracking ? 'translate-x-4' : 'translate-x-1',
                ].join(' ')}
              />
            </button>
          </div>
        </div>
      )}

      <button
        onClick={isStreaming ? onStop : onStart}
        disabled={source === 'video' && !uploadedVideoId && !isStreaming}
        className={[
          'w-full py-3 rounded-xl font-semibold text-white transition-colors text-base',
          isStreaming
            ? 'bg-red-700 hover:bg-red-600'
            : 'bg-green-700 hover:bg-green-600',
          source === 'video' && !uploadedVideoId && !isStreaming
            ? 'opacity-50 cursor-not-allowed'
            : '',
        ].join(' ')}
      >
        {isStreaming ? 'Стоп' : 'Старт'}
      </button>
    </div>
  )
}
