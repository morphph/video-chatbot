'use client'

// Temporarily disabled Clerk authentication for development
// import { SignInButton, SignOutButton, useUser } from '@clerk/nextjs'

export default function AuthHeader() {
  // const { isSignedIn, user } = useUser()
  
  return (
    <div className="flex justify-between items-center mb-8">
      <h1 className="text-4xl font-bold">
        YouTube Video Chatbot
      </h1>
      
      <div className="flex items-center space-x-4">
        <span className="text-sm text-gray-600">
          Development Mode (Authentication Disabled)
        </span>
      </div>
    </div>
  )
}