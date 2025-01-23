#!/bin/python
import asyncio
import sys
import typing_extensions
import logging

import test

async def yield_ex():
    yield 1
    yield 2


async def main():
    typing_extensions.IntVar
    async for a in yield_ex():
        print(a)


if __name__ == '__main__':
    asyncio.run(main())
