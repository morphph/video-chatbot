'use client'

import { useState } from 'react'
import VideoInput from '@/components/video-input'
import ChatInterface from '@/components/chat-interface'

export default function Home() {
  const [videoId, setVideoId] = useState<string | null>(null)
  const [videoMetadata, setVideoMetadata] = useState<any>(null)
  const [isProcessing, setIsProcessing] = useState(false)

  const handleVideoProcessed = (id: string, metadata: any) => {
    setVideoId(id)
    setVideoMetadata(metadata)
    setIsProcessing(false)
  }

  return (
    <main className="flex min-h-screen flex-col items-center justify-start p-8">
      <div className="w-full max-w-4xl">
        <h1 className="text-4xl font-bold text-center mb-8">
          YouTube Video Chatbot
        </h1>
        
        <VideoInput 
          onVideoProcessed={handleVideoProcessed}
          isProcessing={isProcessing}
          setIsProcessing={setIsProcessing}
        />
        
        {videoId && !isProcessing && (
          <ChatInterface 
            videoId={videoId}
            videoMetadata={videoMetadata}
          />
        )}
      </div>
    </main>
  )
}