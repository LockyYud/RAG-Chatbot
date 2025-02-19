'use client';

import { ChatProvider } from '@/libs/context/chat-context';
import { ConfigProvider } from 'antd';

export default function ChatLayout({
    children,
}: Readonly<{
    children: React.ReactNode;
}>) {
    return (
        <ConfigProvider
            theme={{
                components: {
                    Tooltip: {
                        fontSize: 12,
                        fontFamily: 'Noto Sans',
                    },
                },
            }}
        >
            <ChatProvider>{children}</ChatProvider>
        </ConfigProvider>
    );
}
