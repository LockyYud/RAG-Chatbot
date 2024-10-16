'use client';

import * as React from 'react';
import Textarea from 'react-textarea-autosize';
import { useEnterSubmit } from '@/lib/hooks/use-enter-submit';
import { ChatState, Message } from '@/lib/types';
import { useChatContext } from '@/lib/context/chat-context';

export function PromptForm({
    setListMessage,
}: {
    setListMessage: React.Dispatch<React.SetStateAction<Message[]>>;
}) {
    const [input, setInput] = React.useState('');
    const { formRef, onKeyDown } = useEnterSubmit();
    const inputRef = React.useRef<HTMLTextAreaElement>(null);
    const { state } = useChatContext();

    React.useEffect(() => {
        if (inputRef.current) {
            inputRef.current.focus();
        }
    }, []);
    return (
        <form
            ref={formRef}
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            onSubmit={(e: any) => {
                e.preventDefault();
                // Blur focus on mobile
                if (window.innerWidth < 600) {
                    e.target['message']?.blur();
                }

                const value = input.trim();
                if (!value) return;
                const newMessage: Message = { role: 'user', content: value };
                setListMessage((prev) => [...prev, newMessage]);
                setInput('');
            }}
        >
            <Textarea
                disabled={state != ChatState.USER_TURN}
                ref={inputRef}
                tabIndex={0}
                onKeyDown={onKeyDown}
                placeholder="Send a message."
                className=" text-sky-400 min-h-[60px] w-full min-w-full resize-none bg-transparent px-4 py-[1.3rem] focus-within:outline-none sm:text-sm rounded-3xl disabled:cursor-not-allowed disabled:bg-zinc-200 bg-zinc-100 "
                autoFocus
                spellCheck={false}
                autoComplete="off"
                autoCorrect="off"
                name="message"
                rows={1}
                value={input}
                onChange={(e) => setInput(e.target.value)}
            />
        </form>
    );
}
