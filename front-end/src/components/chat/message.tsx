'use client';

import { spinner } from './spinner';
import { cn } from '@/lib/utils';
import RobotOutlined from '@ant-design/icons/lib/icons/RobotOutlined';
import { Message } from '@/lib/types';
import { Card } from 'antd';
import { Dispatch, SetStateAction } from 'react';

// Different types of message bubbles.

export function UserMessage({ children }: { children: React.ReactNode }) {
    return (
        <div className="border-2 rounded-3xl p-1.5 w-fit mr-0 right-0">
            <div className="ml-4 space-y-2 overflow-hidden pr-2">
                {children}
            </div>
        </div>
    );
}

export function BotMessage({ children }: { children: React.ReactNode }) {
    return (
        <div className="group relative flex items-start md:-ml-12">
            <div className="flex size-[25px] shrink-0 select-none items-center justify-center rounded-md border shadow-sm">
                <RobotOutlined />
            </div>
            <div className="ml-4 flex space-y-2 overflow-hidden px-1">
                {children}
            </div>
        </div>
    );
}

export function BotCard({
    children,
    showAvatar = true,
}: {
    children: React.ReactNode;
    showAvatar?: boolean;
}) {
    return (
        <div className="group relative flex items-start md:-ml-12">
            <div
                className={cn(
                    'flex size-[24px] shrink-0 select-none items-center justify-center rounded-md border shadow-sm',
                    !showAvatar && 'invisible',
                )}
            ></div>
            <div className="ml-4 flex-1 pl-2">{children}</div>
        </div>
    );
}

export function SystemMessage({ children }: { children: React.ReactNode }) {
    return (
        <div
            className={
                'mt-2 flex items-center justify-center gap-2 text-xs text-gray-500'
            }
        >
            <div className={'max-w-[600px] flex-initial p-2'}>{children}</div>
        </div>
    );
}

export function SpinnerMessage() {
    return (
        <div className="group relative flex items-start md:-ml-12">
            <div className="flex size-[24px] shrink-0 select-none items-center justify-center rounded-md border bg-primary shadow-sm">
                OpenAI
            </div>
            <div className="ml-4 h-[24px] flex flex-row items-center flex-1 space-y-2 overflow-hidden px-1">
                {spinner}
            </div>
        </div>
    );
}

export function Suggestion({
    suggestMessage,
    setListMessage,
}: {
    suggestMessage: Message[];
    setListMessage: Dispatch<SetStateAction<Message[]>>;
    setSuggestion: Dispatch<SetStateAction<Message[]>>;
}) {
    return (
        <div className="flex gap-2">
            {suggestMessage.map((e, index) => {
                return (
                    <div key={index}>
                        <Card
                            className="cursor-pointer bg-slate-200 hover:bg-slate-300"
                            onClick={() => {
                                setListMessage((prev) => [
                                    ...prev,
                                    { role: 'user', content: e.content },
                                ]);
                            }}
                        >
                            {e.content}
                        </Card>
                    </div>
                );
            })}
        </div>
    );
}
