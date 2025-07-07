'use client'

import { useState } from 'react'
import { processVideo, checkVideoStatus } from '@/lib/api'
import ErrorMessage from './error-message'

interface VideoInputProps {
  onVideoProcessed: (videoId: string, metadata: any) => void
  isProcessing: boolean
  setIsProcessing: (processing: boolean) => void
}

export default function VideoInput({ onVideoProcessed, isProcessing, setIsProcessing }: VideoInputProps) {
  const [url, setUrl] = useState('')
  const [error, setError] = useState('')
  
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    setIsProcessing(true)
    
    try {
      const result = await processVideo(url)
      
      if (result.status === 'processing') {
        // Poll for completion
        const pollInterval = setInterval(async () => {
          const status = await checkVideoStatus(result.video_id)
          
          if (status.status === 'completed') {
            clearInterval(pollInterval)
            onVideoProcessed(result.video_id, status.metadata)
          } else if (status.status === 'failed') {
            clearInterval(pollInterval)
            setError(status.error || 'Failed to process video')
            setIsProcessing(false)
          }
        }, 2000)
      } else if (result.status === 'completed') {
        const status = await checkVideoStatus(result.video_id)
        onVideoProcessed(result.video_id, status.metadata)
      }
    } catch (err: any) {
      setError(err.message || 'Failed to process video')
      setIsProcessing(false)
    }
  }
  
  return (
    <div className="mb-8">
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label htmlFor="url" className="block text-sm font-medium mb-2">
            YouTube Video URL
          </label>
          <input
            type="url"
            id="url"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="https://www.youtube.com/watch?v=..."
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            required
            disabled={isProcessing}
          />
        </div>
        
        {error && (
          <ErrorMessage 
            error={error} 
            onDismiss={() => setError('')}
          />
        )}
        
        <button
          type="submit"
          disabled={isProcessing}
          className="w-full bg-blue-500 text-white py-2 px-4 rounded-lg hover:bg-blue-600 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
        >
          {isProcessing ? 'Processing Video...' : 'Process Video'}
        </button>
      </form>
    </div>
  )
}