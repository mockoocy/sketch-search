FROM node:24-alpine

ENV CI=true
ENV PNPM_HOME="/pnpm"
ENV PATH="$PNPM_HOME:$PATH"
RUN corepack enable

COPY package.json pnpm-lock.yaml /usr/src/app/

WORKDIR /usr/src/app


RUN pnpm install

CMD ["pnpm", "run", "dev", "--host"]
