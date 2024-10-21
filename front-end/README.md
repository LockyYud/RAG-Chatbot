# Front-End

This directory contains the front-end code for the ChatBot UI. It is built using Next.js and includes configurations for ESLint, Tailwind CSS, and PostCSS.

## Project Structure

-   `src/`: Contains the source code for the front-end application.
    -   `components/`: Contains the React components used in the application.
        -   `chat/`: Contains components related to the chat functionality.
            -   `chat-list.tsx`: Defines the `ChatList` component.
            -   `message.tsx`: Defines various message components like `UserMessage`, `BotMessage`, `BotCard`, `SystemMessage`, `SpinnerMessage`, and `Suggestion`.
            -   `empty-screen.tsx`: Defines the `EmptyScreen` component.
    -   `api/`: Contains API-related code.
        -   `chat-api.ts`: Defines the `sendMessages` function to interact with the chat API.
    -   `app/`: Contains the main application layout and pages.
        -   `layout.tsx`: Defines the root layout of the application.
        -   `(public)/page.tsx`: Defines the main page of the application.
-   `public/`: Contains static assets.
-   `styles/`: Contains global styles.
-   `package.json`: Contains the project dependencies and scripts.
-   `tsconfig.json`: TypeScript configuration file.
-   `next.config.mjs`: Next.js configuration file.
-   `.eslintrc.json`: ESLint configuration file.
-   `tailwind.config.ts`: Tailwind CSS configuration file.
-   `postcss.config.mjs`: PostCSS configuration file.

## Getting Started

### Prerequisites

-   Node.js (version 14.x or later)
-   npm (version 6.x or later) or yarn (version 1.x or later)

### Installation

1. Navigate to the `front-end` directory:

    ```sh
    cd front-end
    ```

2. Install the dependencies:

    ```sh
    npm install
    ```

    or if you are using yarn:

    ```sh
    yarn install
    ```

### Running the Development Server

To start the development server, run:

```sh
npm run dev
```
