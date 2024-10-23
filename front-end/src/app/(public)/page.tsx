"use client";
import { Chat } from "@/components/chat/chat";
import ConfigBE from "@/components/chat/config";
import { MenuFoldOutlined, MenuUnfoldOutlined } from "@ant-design/icons";
import { Button, Drawer, Layout } from "antd";
import { Content, Header } from "antd/es/layout/layout";
import { useState } from "react";

export default function Page() {
    const [open, setOpen] = useState(false);

    const showDrawer = () => {
        setOpen(true);
    };

    const onClose = () => {
        setOpen(false);
    };
    return (
        <div>
            <Layout className="h-screen bg-background">
                <Drawer
                    title="Config"
                    onClose={onClose}
                    open={open}
                    placement="left"
                >
                    <ConfigBE />
                </Drawer>
                <Header style={{ padding: 0 }} className="bg-background fixed">
                    <Button
                        type="text"
                        icon={
                            open ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />
                        }
                        onClick={showDrawer}
                        style={{
                            fontSize: "16px",
                            width: 64,
                            height: 64,
                        }}
                    />
                </Header>
                <Content
                    style={{
                        // margin: '24px 16px',
                        minHeight: 280,
                    }}
                    className="bg-background"
                >
                    <Chat></Chat>
                </Content>
            </Layout>
        </div>
    );
}
