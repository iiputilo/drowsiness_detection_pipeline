import React, { useState, useCallback } from 'react'
import { useWebSocket } from './hooks/useWebSocket'
import PipelineSelector from './components/PipelineSelector'
import VideoDisplay from './components/VideoDisplay'
import ControlPanel from './components/ControlPanel'
import StatusPanel from './components/StatusPanel'

export default function App() {
  const [pipeline, setPipeline] = useState<'yolo' | 'mediapipe'>('yolo')
  const [source, setSource] = useState<'webcam' | 'video'>('webcam')
  const [showOverlay, setShowOverlay] = useState(true)
  const [gazeTracking, setGazeTracking] = useState(false)
  const [uploadedVideoId, setUploadedVideoId] = useState<string | null>(null)

  const ws = useWebSocket()

  const handleStart = useCallback(() => {
    ws.connect({
      source,
      pipeline,
      show_overlay: showOverlay,
      gaze_tracking: gazeTracking,
      video_id: source === 'video' ? uploadedVideoId : null,
    })
  }, [ws, source, pipeline, showOverlay, gazeTracking, uploadedVideoId])

  const handleStop = useCallback(() => {
    ws.disconnect()
  }, [ws])

  const handlePipelineChange = useCallback((p: 'yolo' | 'mediapipe') => {
    setPipeline(p)
  }, [])

  return (
    <div className="min-h-screen bg-gray-950 text-white">
      <header className="bg-gray-900 border-b border-gray-800 px-6 py-3 flex items-center gap-3">
        <div className="w-2.5 h-2.5 rounded-full bg-blue-500" />
        <h1 className="text-lg font-semibold">Система оценки когнитивного состояния и точки внимания человека</h1>
      </header>

      <main className="p-4 grid grid-cols-1 lg:grid-cols-3 gap-4 max-w-7xl mx-auto">
        <div className="lg:col-span-2 flex flex-col gap-4">
          <VideoDisplay
            pipeline={pipeline}
            mainFrame={ws.mainFrame}
            faceFrame={ws.faceFrame}
            isStreaming={ws.isStreaming}
          />
        </div>

        <div className="flex flex-col gap-4">
          <PipelineSelector
            pipeline={pipeline}
            onChange={handlePipelineChange}
            disabled={ws.isStreaming}
          />

          <StatusPanel
            currentState={ws.currentState}
            alarm={ws.alarm}
            metrics={ws.metrics}
            pipeline={pipeline}
            alarmCount={ws.alarmCount}
          />

          <ControlPanel
            source={source}
            onSourceChange={setSource}
            pipeline={pipeline}
            showOverlay={showOverlay}
            onShowOverlayChange={setShowOverlay}
            gazeTracking={gazeTracking}
            onGazeTrackingChange={setGazeTracking}
            isConnected={ws.isConnected}
            isStreaming={ws.isStreaming}
            isPaused={ws.isPaused}
            uploadedVideoId={uploadedVideoId}
            onVideoUploaded={setUploadedVideoId}
            onStart={handleStart}
            onStop={handleStop}
            onPause={() => ws.sendControl('pause')}
            onResume={() => ws.sendControl('resume')}
            onReset={() => ws.sendControl('reset')}
          />
        </div>
      </main>
    </div>
  )
}
