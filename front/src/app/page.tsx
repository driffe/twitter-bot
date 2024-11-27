'use client';

import dynamic from 'next/dynamic';
import { TwitterBotDetector } from './components/twitter-bot-detector';

const TwitterBotDetectorClient = dynamic(
  () => Promise.resolve(TwitterBotDetector),
  { ssr: false }
);

export default function Page() {
  return <TwitterBotDetectorClient />;
}