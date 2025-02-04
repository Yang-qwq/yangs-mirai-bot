# -*- coding: utf-8 -*-

import time

import main

msg_list = [
    {"syncId": "", "data": {"code": 0, "session": "SESSION"}},
    {"syncId": "-1",
     "data": {"type": "GroupMessage", "messageChain": [{"type": "Source", "id": 1, "time": int(time.time())}],
              "sender": {"id": 10001, "memberName": "Alice", "specialTitle": "",
                         "permission": "MEMBER", "joinTimestamp": 0, "lastSpeakTimestamp": int(time.time()),
                         "muteTimeRemaining": 0,
                         "group": {"id": 114514, "name": "机器人测试群",
                                   "permission": "MEMBER"}}}},  # 普通群成员说话
    {"syncId": "-1",
     "data": {"type": "GroupRecallEvent", "authorId": 10001, "messageId": 1, "time": int(time.time()),
              "group": {"id": 114514, "name": "机器人测试群", "permission": "MEMBER"},
              "operator": {"id": 10001, "memberName": "Alice", "specialTitle": "",
                           "permission": "ADMINISTRATOR", "joinTimestamp": 0,
                           "lastSpeakTimestamp": int(time.time()), "muteTimeRemaining": 0,
                           "group": {"id": 114514, "name": "机器人测试群", "permission": "MEMBER"}}}},  # 消息撤回
    {"syncId": "-1", "data": {"type": "NudgeEvent", "fromId": 10001, "target": 10002,
                              "subject": {"kind": "Group", "id": 114514, "name": "机器人测试群",
                                          "permission": "OWNER"}, "action": "戳了戳", "suffix": ""}},  # 群内戳一戳
    {"syncId": "-1", "data": {"type": "NudgeEvent", "fromId": 10001, "target": 10002,
                              "subject": {"kind": "Friend", "id": 3484861532, "nickname": "Alice", "remark": "爱丽丝"},
                              "action": "戳了戳", "suffix": ""}}  # 私聊戳一戳
]


class TestClassMiraiResponse:
    def test_sync_id(self):
        for msg in msg_list:
            mirai_res = main.MiraiResponse(msg)
            assert mirai_res.sync_id == msg['syncId']

    def test_message_chain_get(self):
        for msg in msg_list:
            mirai_res = main.MiraiResponse(msg)
            if msg['data'].get('messageChain', None):  # 判断是否带有messageChain
                assert mirai_res.get_message_chain() == msg['data']['messageChain']
            else:
                assert mirai_res.get_message_chain() == []

    def test_message_type_get(self):
        for msg in msg_list:
            mirai_res = main.MiraiResponse(msg)

            # 是否拥有type字段
            if 'type' not in msg['data']:
                assert mirai_res.get_message_type() is None
            else:
                assert mirai_res.get_message_type() == msg['data']['type']

    def test_sender_id_get(self):
        for msg in msg_list:
            mirai_res = main.MiraiResponse(msg)

            # 获取消息类型
            if 'type' not in msg['data']:
                assert mirai_res.get_sender_id() is None
            else:
                message_type = msg['data']['type']
                if message_type in ['GroupMessage', 'FriendMessage']:  # 最常见的类型
                    assert mirai_res.get_sender_id() == msg['data']['sender']['id']
                elif message_type == 'GroupRecallEvent':  # 群聊撤回事件
                    assert mirai_res.get_sender_id() == msg['data']['operator']['id']
                elif message_type == 'NudgeEvent':  # 戳一戳事件
                    assert mirai_res.get_sender_id() == msg['data']['fromId']

    # def test_group_id_get(self):
    #     for msg in msg_list:
    #         mirai_res = main.MiraiResponse(msg)
    #
    #         if 'type' not in msg['data']:
    #             assert mirai_res.get_group_id() is None
    #
    #             # 不符合条件，跳过测试
    #             continue
    #         else:
    #             message_type = msg['data']['type']
    #
    #         if message_type in ['GroupMessage', 'GroupRecallEvent']:
    #             assert mirai_res.get_group_id() == msg['data']['group']['id']
    #         elif message_type == 'NudgeEvent':
    #             assert mirai_res.get_group_id() == msg['data']['subject']['id']
    #         else:
    #             assert mirai_res.get_group_id() is None

    def test_sender_name_get(self):
        for msg in msg_list:
            mirai_res = main.MiraiResponse(msg)

            if 'type' not in msg['data']:
                assert mirai_res.get_sender_name() == (None, None)

                # 不符合条件，跳过测试
                continue
            else:
                message_type = msg['data']['type']

            if message_type == 'GroupMessage':
                assert msg['data']['sender']['memberName'], msg['data']['sender']['specialTitle']
            elif message_type == 'FriendMessage':
                assert msg['data']['sender']['nickname'], msg['data']['sender']['remark']
            elif message_type == 'GroupRecallEvent':
                assert msg['data']['operator']['memberName'], msg['data']['operator']['specialTitle']
            else:
                assert mirai_res.get_sender_name() == (None, None)
