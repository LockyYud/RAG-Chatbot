import { cn } from "@/libs/utils";
import { SendOutlined } from "@ant-design/icons";
import { Button, ConfigProvider, Tooltip } from "antd";

interface SendButtonProps extends React.ComponentProps<"button"> {
    state: boolean;
}

export const SendButton = ({
    className,
    onSubmit,
    disabled,
    state,
}: SendButtonProps) => {
    return (
        <ConfigProvider
            theme={{
                components: {
                    Button: {
                        defaultHoverBg: state ? "#494949" : "",
                        defaultHoverBorderColor: "#ffffff",
                        textHoverBg: "#ffffff",
                        textTextColor: "#ffffff",
                        textTextHoverColor: "#ffffff",
                    },
                },
            }}
        >
            <Tooltip
                color="#769bdb"
                title={state ? "Send message" : "Sending..."}
            >
                <Button
                    onSubmit={onSubmit}
                    disabled={disabled}
                    className={cn(
                        "object-center bg-black text-white hover:bg-slate-800",
                        className
                    )}
                    shape="circle"
                    htmlType="submit"
                    size="large"
                >
                    <SendOutlined />
                </Button>
            </Tooltip>
        </ConfigProvider>
    );
};
