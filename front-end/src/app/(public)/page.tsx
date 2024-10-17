"use client";
import { Chat } from "@/components/chat/chat";
import ConfigBE from "@/components/chat/config";
import { MenuFoldOutlined, MenuUnfoldOutlined } from "@ant-design/icons";
import { Button, Layout, theme } from "antd";
import { Content, Header } from "antd/es/layout/layout";
import Sider from "antd/es/layout/Sider";
import { useState } from "react";

export default function Page() {
    const [collapsed, setCollapsed] = useState(false);
    const {
        token: { colorBgContainer, borderRadiusLG },
    } = theme.useToken();
    // useEffect(() => {
    //     // eslint-disable-next-line @typescript-eslint/no-unused-vars
    //     const handleBeforeUnload = async (event: BeforeUnloadEvent) => {
    //         await deleteConversation(id);
    //     };

    //     // Attach the event listener
    //     window.addEventListener("beforeunload", handleBeforeUnload);

    //     // Cleanup the event listener
    //     return () => {
    //         window.removeEventListener("beforeunload", handleBeforeUnload);
    //     };
    // }, [id]);
    return (
        <div>
            <Layout>
                <Sider
                    trigger={null}
                    collapsible
                    collapsed={collapsed}
                    collapsedWidth={0}
                    className="bg-transparent"
                >
                    <ConfigBE />
                </Sider>
                <Layout>
                    <Header
                        style={{ padding: 0, background: colorBgContainer }}
                    >
                        <Button
                            type="text"
                            icon={
                                collapsed ? (
                                    <MenuUnfoldOutlined />
                                ) : (
                                    <MenuFoldOutlined />
                                )
                            }
                            onClick={() => setCollapsed(!collapsed)}
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
                            padding: 24,
                            minHeight: 280,
                            background: colorBgContainer,
                            borderRadius: borderRadiusLG,
                        }}
                    >
                        <Chat></Chat>
                    </Content>
                </Layout>
            </Layout>
        </div>
    );
}
