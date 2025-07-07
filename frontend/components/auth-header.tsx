'use client'

import { SignInButton, SignOutButton, useUser } from '@clerk/nextjs'

export default function AuthHeader() {
  const { isSignedIn, user } = useUser()
  
  return (
    <div className="flex justify-between items-center mb-8">
      <h1 className="text-4xl font-bold">
        YouTube Video Chatbot
      </h1>
      
      <div className="flex items-center space-x-4">
        {isSignedIn ? (
          <>
            <span className="text-sm text-gray-600">
              Welcome, {user?.firstName || user?.emailAddresses[0]?.emailAddress}
            </span>
            <SignOutButton>
              <button className="bg-red-500 text-white px-4 py-2 rounded-lg hover:bg-red-600 transition-colors">
                Sign Out
              </button>
            </SignOutButton>
          </>
        ) : (
          <SignInButton>
            <button className="bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600 transition-colors">
              Sign In
            </button>
          </SignInButton>
        )}
      </div>
    </div>
  )
}