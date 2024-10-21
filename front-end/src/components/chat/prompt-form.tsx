"use client";

import * as React from "react";
import Textarea from "react-textarea-autosize";
import { useEnterSubmit } from "@/libs/hooks/use-enter-submit";
import { Message } from "@/libs/types";
import { Button } from "antd";
import { SendOutlined } from "@ant-design/icons";

export function PromptForm({
    setListMessage,
}: {
    setListMessage: React.Dispatch<React.SetStateAction<Message[]>>;
}) {
    const [input, setInput] = React.useState("");
    const { formRef, onKeyDown } = useEnterSubmit();
    const inputRef = React.useRef<HTMLTextAreaElement>(null);

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
                    e.target["message"]?.blur();
                }

                const value = input.trim();
                if (!value) return;
                const newMessage: Message = { role: "user", content: value };
                setListMessage((prev) => [...prev, newMessage]);
                setInput("");
            }}
            className="mb-4"
        >
            <div className="relative">
                <Textarea
                    ref={inputRef}
                    tabIndex={0}
                    onKeyDown={onKeyDown}
                    placeholder="Send a message."
                    className=" text-black min-h-[60px] w-full bg-transparent px-4 py-[1.3rem] focus-within:outline-none sm:text-base rounded-3xl disabled:cursor-not-allowed disabled:bg-zinc-200 bg-zinc-100 "
                    autoFocus
                    spellCheck={false}
                    autoComplete="off"
                    autoCorrect="off"
                    name="message"
                    rows={1}
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                />
                <div className="absolute my-auto top-0 h-full flex flex-col align-middle justify-center right-3">
                    <Button
                        className=" object-center bg-black text-white hover:bg-slate-800"
                        shape="circle"
                        htmlType="submit"
                        size="large"
                    >
                        <SendOutlined />
                    </Button>
                </div>
            </div>
        </form>
    );
}
