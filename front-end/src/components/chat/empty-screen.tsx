import Link from 'next/link';

export function EmptyScreen() {
    return (
        <div className="mx-auto max-w-2xl px-4 ">
            <div className="flex flex-col gap-2 rounded-lg border p-8 bg-blue-100">
                <h1 className="text-lg font-semibold">
                    Welcome to Sao Khue AI Chatbot!
                </h1>
                <p className="leading-normal text-muted-foreground">
                    This is a chatbot designed to help with learning English,
                    with topics and grades identified.
                </p>
                <p className="leading-normal text-muted-foreground">
                    If you want to use our other AI tools, please go to the{' '}
                    <Link className="text-blue-500" href={'/'}>
                        Sao Khue AI
                    </Link>
                    .
                </p>
            </div>
        </div>
    );
}
