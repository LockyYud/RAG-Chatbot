"use client";

import { spinner } from "./spinner";
import { cn } from "@/libs/utils";
import RobotOutlined from "@ant-design/icons/lib/icons/RobotOutlined";
import MarkdownRenderer from "../MarkdownRender/markdown";

// Different types of message bubbles.

export function UserMessage({ children }: { children: React.ReactNode }) {
    return (
        <div className="bg-blue-500 text-white rounded-3xl p-3 w-fit max-w-[80%] mr-0 right-0 shadow-md ml-auto">
            <div className="overflow-hidden whitespace-pre-line">
                {children}
            </div>
        </div>
    );
}

export function BotMessage({ children }: { children: React.ReactNode }) {
    return (
        <div className="group relative flex items-start mb-2 mr-auto max-w-[80%]">
            <div className="flex size-[25px] shrink-0 select-none items-center justify-center rounded-full border bg-gray-200 shadow-sm ">
                <RobotOutlined />
            </div>
            <div className="ml-3 flex-1 p-3 bg-gray-100 rounded-3xl shadow-md whitespace-pre-line">
                {typeof children === "string" ? (
                    <MarkdownRenderer markdownText={children} />
                ) : (
                    children
                )}
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
        <div className="group relative flex items-start md:-ml-12 mb-2">
            <div
                className={cn(
                    "flex size-[24px] shrink-0 select-none items-center justify-center rounded-full border bg-gray-200 shadow-sm",
                    !showAvatar && "invisible"
                )}
            >
                {/* Optionally, you can include an avatar here */}
            </div>
            <div className="ml-3 flex-1 p-3 bg-gray-100 rounded-3xl shadow-md whitespace-pre-line">
                {children}
            </div>
        </div>
    );
}

export function SystemMessage({ children }: { children: React.ReactNode }) {
    return (
        <div className="mt-2 flex items-center justify-center gap-2 text-xs text-gray-500">
            <div className="max-w-[600px] flex-initial p-2 bg-gray-200 rounded-lg shadow-sm whitespace-pre-line">
                {children}
            </div>
        </div>
    );
}

export function SpinnerMessage() {
    return (
        <div className="group relative flex items-start md:-ml-12 mb-2">
            <div className="flex size-[24px] shrink-0 select-none items-center justify-center rounded-full border bg-primary shadow-sm">
                OpenAI
            </div>
            <div className="ml-3 h-[24px] flex flex-row items-center flex-1 space-y-2 overflow-hidden px-1 whitespace-pre-line">
                {spinner}
            </div>
        </div>
    );
}
