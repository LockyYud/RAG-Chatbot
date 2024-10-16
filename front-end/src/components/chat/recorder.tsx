import { cn } from '@/lib/utils';
import { AudioOutlined } from '@ant-design/icons';
import { Button, ConfigProvider, Tooltip } from 'antd';

interface TextRecordProps extends React.ComponentProps<'div'> {
    text: string;
    funcSend: () => void;
    funcCancel: () => void;
}
export function TextRecord({
    className,
    text,
    funcSend,
    funcCancel,
}: TextRecordProps) {
    return (
        <div className={cn('bg-background mb-3', className)}>
            <div className=" grid grid-cols-8 gap-3 px-4 bg-slate-100 min-h-18 h-14 content-center rounded-3xl">
                <span className="col-span-6 flex flex-col justify-center h-full ">
                    {text}
                </span>
                <Button shape="round" className="primary" onClick={funcSend}>
                    Send
                </Button>
                <Button shape="round" onClick={funcCancel}>
                    Cancel
                </Button>
            </div>
        </div>
    );
}

interface RecordButtonProps extends React.ComponentProps<'button'> {
    state: boolean;
    startFunc: () => void;
    stopFunc: () => void;
}

export const RecordButton = ({
    disabled,
    state,
    startFunc,
    stopFunc,
}: RecordButtonProps) => {
    return (
        <div className="absolute right-0 flex flex-col justify-center h-full p-3">
            <ConfigProvider
                theme={{
                    components: {
                        Button: {
                            defaultHoverBg: state ? 'red' : '',
                            defaultHoverBorderColor: 'transparent',
                        },
                    },
                }}
            >
                <Tooltip
                    color="red"
                    title={state ? 'Stop record' : 'Start record'}
                >
                    <Button
                        disabled={disabled}
                        onClick={() => {
                            if (state) {
                                stopFunc();
                            } else {
                                startFunc();
                            }
                        }}
                        size={!state ? 'middle' : 'large'}
                        className={cn(
                            state ? 'bg-red-500 hover:bg-red-500' : '',
                        )}
                        shape="circle"
                    >
                        <AudioOutlined
                            style={{
                                color: state ? 'white' : 'red',
                            }}
                        />
                    </Button>
                </Tooltip>
            </ConfigProvider>
        </div>
    );
};
